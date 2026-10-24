"""单步视觉闭环编排。"""

from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .artifacts import ArtifactWriter
from .astra.client import Halt
from .astra.policy import PhaseState, PolicyValidator
from .config import AppConfig
from .robot.ik import Q_MAX, Q_MIN, FirmwareIK
from .robot.safety import SafetyChecker, SafetyError
from .robot.sim import SimBackend
from .robot.task import PickPlaceTask
from .rotations import quat_wxyz_to_matrix
from .types import ExecutionResult, Pose, RolloutResult, VerificationResult


def build_observation(
    cfg: AppConfig, obs, phase: str, recent: list[dict], *, visual_candidates=None
) -> dict:
    s = cfg.safety
    return {
        "task": (
            "Pick up the red 30 mm cube and place it inside the green circular "
            "target zone on the table."
        ),
        "phase": phase,
        "frame_id": obs.frame_id,
        "commanded_tcp": obs.commanded_tcp.as_dict(),
        "measured_tcp": obs.arm.tcp.as_dict(),
        "flange": obs.arm.flange.as_dict(),
        "joints": [float(v) for v in obs.arm.q],
        "gripper": {
            "opening_mm": float(obs.gripper.opening_mm),
            "closure": float(obs.gripper.closure),
        },
        "tool": {
            "tcp_offset_m": float(cfg.tcp_offset_m),
            "finger_open_mm": float(cfg.gripper.open_m * 1000.0),
            "finger_direction": "+Z (from mounting flange to TCP)",
            "finger_axis": "tool X",
        },
        "workspace": {
            "x": list(s.workspace_x),
            "y": list(s.workspace_y),
            "z": list(s.workspace_z),
        },
        "limits": {
            "max_translation_m": float(s.max_step_m),
            "max_translation_near_grasp_m": float(s.max_step_near_grasp_m),
            "max_rotation_rad": float(s.max_rotation_rad),
            "max_rotation_near_grasp_rad": float(s.max_rotation_near_grasp_rad),
        },
        "target_zone": {
            "center_xy": [float(v) for v in cfg.scene.target_center],
            "radius_m": float(cfg.scene.target_radius_m),
        },
        "visual_primary": "head",
        "visual_candidates": visual_candidates or {},
        "visual_note": (
            "red_cube_table_xy / green_zone_table_xy are approximate single-view "
            "intersections of the detected color blob with the table plane; "
            "they are not depth or a precision grasp pose. Use ONLY the head "
            "candidates for XY alignment; the wrist camera moves with the gripper."
        ),
        "recent_executed": recent[-5:],
    }


class Rollout:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        policy,
        out_dir: str | Path,
        seed: int = 0,
        mode: str = "scripted",
        viewer: bool = False,
        ik: FirmwareIK | None = None,
        backend_cls=None,
        realtime: bool | None = None,
        on_step_start=None,
        on_step_done=None,
        sim=None,
        speed: float = 1.0,
    ):
        self.cfg = cfg
        self.policy = policy
        self.out_dir = Path(out_dir)
        self.seed = int(seed)
        self.mode = mode
        self.viewer = viewer
        self.ik = ik or FirmwareIK.from_source()
        self.backend_cls = backend_cls or SimBackend
        self.realtime = realtime
        self.on_step_start = on_step_start
        self.on_step_done = on_step_done
        self._shared_sim = sim
        self.speed = float(speed)
        self.safety = SafetyChecker(cfg.safety)
        margin = cfg.safety.joint_margin_rad
        self.q_min = Q_MIN + margin
        self.q_max = Q_MAX - margin
        self.validator = PolicyValidator(self.safety, confidence_min=cfg.astra.confidence_min)
        self.phase = PhaseState()
        self.artifacts = ArtifactWriter(self.out_dir)
        self._last_block = None
        self._last_images = None

    # ------------------------------------------------------------------ helpers
    def _tcp_to_flange(self, tcp: Pose) -> tuple[np.ndarray, np.ndarray]:
        R = quat_wxyz_to_matrix(tcp.quaternion_wxyz)
        offset = R @ np.array([0.0, 0.0, self.cfg.tcp_offset_m])
        return np.asarray(tcp.position, dtype=float) - offset, R

    def _residual(self, q, pos_d, R_d) -> tuple[float, float]:
        p, R = self.ik.fk(q)
        pos_err = float(np.linalg.norm(p - pos_d))
        rot_err = float(np.linalg.norm(R - R_d))
        return pos_err, rot_err

    def _move(self, sim: SimBackend, current_obs, decision) -> tuple[ExecutionResult, dict]:
        pos_d, R_d = self._tcp_to_flange(decision.tcp_target)
        seed = np.asarray(current_obs.arm.q, dtype=float)
        q, ok = self.ik.solve_local(
            pos_d,
            R_d,
            seed,
            self.q_min,
            self.q_max,
            pos_tol=self.cfg.safety.ik_pos_tol_m,
            rot_tol=self.cfg.safety.ik_rot_tol_rad,
            max_joint_jump=self.cfg.safety.max_joint_jump_rad,
        )
        pos_err, rot_err = self._residual(q, pos_d, R_d)
        info = {
            "q_seed": seed.tolist(),
            "q_solution": q.tolist(),
            "pos_error_m": pos_err,
            "rot_error_rad": rot_err,
            "joint_jump_rad": float(np.max(np.abs(q - seed))),
        }
        if not ok:
            raise SafetyError(f"IK failed: {pos_err:.5f} m / {rot_err:.5f} rad")
        self.safety.check_ik(seed, q, pos_error_m=pos_err, rot_error_rad=rot_err)
        self.safety.check_path(current_obs.arm.tcp, decision.tcp_target)
        step_distance = float(
            np.linalg.norm(
                np.asarray(decision.tcp_target.position) - np.asarray(current_obs.arm.tcp.position)
            )
        )
        pos_tol = max(0.002, min(0.008, 0.30 * step_distance))
        result = sim.execute_joints(q, tcp_target=decision.tcp_target, pos_tolerance_m=pos_tol)
        return result, info

    def _close_and_lift(self, sim: SimBackend, obs) -> ExecutionResult:
        before = sim.block_position().copy()
        sim.close_gripper()
        sim.step(int(self.cfg.gripper.close_time_s / self.cfg.scene.timestep))
        after_close = sim.block_position().copy()
        target = np.asarray(sim.tcp_position(), dtype=float) + np.array(
            [0.0, 0.0, self.cfg.gripper.trial_lift_m]
        )
        q_seed = np.asarray(sim.q(), dtype=float)
        pos_d, R_d = self._tcp_to_flange(
            Pose(tuple(float(v) for v in target), obs.commanded_tcp.quaternion_wxyz)
        )
        q, ok = self.ik.solve_local(
            pos_d,
            R_d,
            q_seed,
            self.q_min,
            self.q_max,
            pos_tol=self.cfg.safety.ik_pos_tol_m,
            rot_tol=self.cfg.safety.ik_rot_tol_rad,
            max_joint_jump=self.cfg.safety.max_joint_jump_rad,
        )
        if not ok:
            return ExecutionResult(False, False, "trial lift IK failed", 0.0)
        result = sim.execute_joints(
            q,
            tcp_target=Pose(tuple(float(v) for v in target), obs.commanded_tcp.quaternion_wxyz),
            pos_tolerance_m=0.002,
        )
        if not result.ok:
            return result
        sim.step(int(0.2 / self.cfg.scene.timestep))
        after_lift = sim.block_position().copy()
        rose = float(after_lift[2] - before[2])
        xy_err = float(np.linalg.norm(after_lift[:2] - after_close[:2]))
        if rose < 0.002:
            return ExecutionResult(False, False, f"block did not follow lift ({rose:.4f} m)", 0.0)
        if xy_err > 0.02:
            return ExecutionResult(False, False, f"block drifted during lift ({xy_err:.4f} m)", 0.0)
        return result

    # ---------------------------------------------------------------------- run
    def run(self, max_steps: int | None = None) -> RolloutResult:
        started = time.monotonic()
        max_steps = int(max_steps or self.cfg.max_steps)
        self.artifacts.write_config(
            self.cfg,
            astra_info={
                "mode": self.mode,
                "key_present": bool(getattr(self.policy, "key_present", False)),
            },
        )
        recent: list[dict] = []
        errors: list[str] = []
        step = 0
        verification = None
        try:
            manager = (
                nullcontext(self._shared_sim)
                if self._shared_sim is not None
                else self.backend_cls(
                    self.cfg,
                    viewer=self.viewer,
                    realtime=self.realtime,
                    speed=self.speed,
                )
            )
            with manager as sim:
                task = PickPlaceTask(self.cfg, sim, self.safety)
                task.reset(self.seed)
                for step in range(max_steps):
                    if self.on_step_start is not None:
                        self.on_step_start(step, self.phase.phase)
                    obs = sim.observe(frame_id=step, phase=self.phase.phase)
                    raw = None
                    phase_before = self.phase.phase
                    obs_json = build_observation(
                        self.cfg,
                        obs,
                        self.phase.phase,
                        recent,
                        visual_candidates=task.visual_candidates(obs.images),
                    )
                    views = [("head", obs.images["head"]), ("wrist", obs.images["wrist"])]
                    before_paths = {
                        name: self.artifacts.save_image(f"step_{step:04d}_before_{name}.jpg", image)
                        for name, image in obs.images.items()
                    }
                    try:
                        raw = self.policy.propose(obs, sim, self.phase.phase, obs_json, views)
                        self.artifacts.replay(raw)
                        decision = self.validator.validate(
                            raw,
                            current=obs.commanded_tcp,
                            phase=self.phase.phase,
                            frame_id=step,
                            gripper_closed=obs.gripper.closure >= 0.5,
                            block_position=sim.block_position(),
                            target_center=self.cfg.scene.target_center,
                            target_z=0.0,
                        )
                    except (Halt, SafetyError) as exc:
                        self.phase.set_held()
                        errors.append(str(exc))
                        self.artifacts.step(
                            {
                                "step": step,
                                "frame_id": step,
                                "phase_before": phase_before,
                                "phase_after": self.phase.phase,
                                "proposal": raw,
                                "decision": None,
                                "executed": {"ok": False, "reached": False, "error": str(exc)},
                                "images": before_paths,
                            }
                        )
                        break

                    ik_info = None
                    try:
                        if decision.status == "move":
                            executed, ik_info = self._move(sim, obs, decision)
                            self.artifacts.ik(
                                {
                                    "step": step,
                                    "target_tcp": decision.tcp_target.as_dict(),
                                    **(ik_info or {}),
                                    "ok": executed.ok,
                                }
                            )
                        elif decision.status == "grasp":
                            if self.phase.phase == "approach":
                                executed = ExecutionResult(True, True, None, 0.0)
                            else:
                                executed = self._close_and_lift(sim, obs)
                        elif decision.status == "release":
                            sim.open_gripper()
                            sim.step(int(0.6 / self.cfg.scene.timestep))
                            executed = ExecutionResult(True, True, None, 1.0)
                        else:
                            executed = ExecutionResult(True, False, None, 0.0)
                    except (Halt, SafetyError) as exc:
                        self.phase.set_held()
                        errors.append(str(exc))
                        executed = ExecutionResult(False, False, str(exc), 0.0)

                    if not executed.ok:
                        self.phase.set_held()
                        if executed.error:
                            errors.append(executed.error)
                    else:
                        self.phase.advance(decision, executed_ok=executed.ok)

                    after_paths = {}
                    try:
                        after_obs = sim.observe(frame_id=step, phase=self.phase.phase)
                        after_paths = {
                            name: self.artifacts.save_image(
                                f"step_{step:04d}_after_{name}.jpg", image
                            )
                            for name, image in after_obs.images.items()
                        }
                    except Exception as exc:  # 图像失败不阻断结果
                        errors.append(f"after image failed: {exc}")

                    self.artifacts.step(
                        {
                            "step": step,
                            "frame_id": step,
                            "phase_before": phase_before,
                            "phase_after": self.phase.phase,
                            "proposal": raw,
                            "decision": {
                                "status": decision.status,
                                "tcp_target": decision.tcp_target.as_dict(),
                                "gripper": decision.gripper,
                                "clamped": decision.clamped,
                                "reason": decision.reason,
                                "evidence": decision.evidence,
                            },
                            "executed": {
                                "ok": executed.ok,
                                "reached": executed.reached,
                                "error": executed.error,
                                "elapsed_s": executed.elapsed_s,
                            },
                            "ik": ik_info,
                            "images": {**before_paths, **after_paths},
                        }
                    )
                    if self.on_step_done is not None:
                        self.on_step_done(
                            step,
                            phase_before,
                            self.phase.phase,
                            decision.status,
                            executed.ok,
                            executed.error,
                        )
                    if self.phase.phase in ("done", "held"):
                        break

                verification = task.verify()
                if verification.success and self.phase.phase == "released":
                    self.phase.phase = "done"
                if not verification.success:
                    errors.append(f"verification: {verification.reason}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            verification = verification or None

        elapsed = time.monotonic() - started
        success = bool(verification and verification.success and self.phase.phase == "done")
        result = RolloutResult(
            success=success,
            score=float(verification.score) if verification else 0.0,
            steps=step + 1,
            phase=self.phase.phase,
            verification=verification or VerificationResult(False, 0.0, "no verification"),
            errors=tuple(errors),
            elapsed_s=elapsed,
            seed=self.seed,
            mode=self.mode,
        )
        self.artifacts.finish(result)
        return result
