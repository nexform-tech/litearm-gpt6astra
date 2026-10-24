"""确定性 MuJoCo lock-step 后端。

不启动后台线程，不使用墙钟阻塞运动；每个 ``step()`` 推进一个物理步。
这样测试和回放都是可复现的。可选的 viewer 只负责显示。
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import mujoco
import numpy as np

from ..config import AppConfig
from ..rotations import matrix_to_quat_wxyz
from ..types import ArmState, ExecutionResult, GripperState, Observation, Pose
from .cameras import Cameras


class SimBackend:
    def __init__(
        self,
        cfg: AppConfig,
        *,
        viewer: bool = False,
        realtime: bool | None = None,
        speed: float = 1.0,
    ):
        self.cfg = cfg
        model_path = Path(cfg.scene.model_path)
        if not model_path.is_file():
            raise FileNotFoundError(
                f"{model_path} missing; run `python scripts/build_sim_model.py`"
            )
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.model.opt.timestep = cfg.scene.timestep
        self.data = mujoco.MjData(self.model)
        self.cameras = Cameras(self.model, self.data)

        self._arm_acts = [self.model.actuator(f"j{i}").id for i in range(1, 8)]
        self._finger_acts = [
            self.model.actuator("finger_right_act").id,
            self.model.actuator("finger_left_act").id,
        ]
        self._arm_joints = [self.model.joint(f"j{i}").id for i in range(1, 8)]
        self._finger_joints = [
            self.model.joint("finger_right").id,
            self.model.joint("finger_left").id,
        ]
        self._block_joint = self.model.joint("block_free").id
        self._gripper_base = self.model.body("gripper_base").id
        self._block_body = self.model.body("block").id
        self._weld_id = self.model.equality("grasp_weld").id if self.model.neq else None

        self._commanded_q = np.asarray(cfg.home_q, dtype=float)
        self._viewer = None
        self._viewer_counter = 0
        self._realtime = bool(viewer) if realtime is None else bool(realtime)
        self._speed = max(0.1, float(speed))
        self._sync_every = max(4, int(round(8 * self._speed)))
        self._next_tick = time.perf_counter()
        if viewer:
            from mujoco import viewer as mj_viewer  # 子模块必须显式导入

            self._viewer = mj_viewer.launch_passive(self.model, self.data)

    # ---------------------------------------------------------------- lifecycle
    def __enter__(self) -> SimBackend:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.cameras.close()
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    # ------------------------------------------------------------------- reset
    def reset(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._next_tick = time.perf_counter()
        mujoco.mj_resetData(self.model, self.data)
        home = np.asarray(self.cfg.home_q, dtype=float)
        self.data.qpos[:7] = home
        adr = int(self.model.jnt_qposadr[self._block_joint])
        self.data.qpos[adr] = rng.uniform(*self.cfg.scene.start_x)
        self.data.qpos[adr + 1] = rng.uniform(*self.cfg.scene.start_y)
        self.data.qpos[adr + 2] = self.cfg.scene.block_size_m / 2.0
        self.data.qpos[adr + 3 : adr + 7] = [1.0, 0.0, 0.0, 0.0]
        for jid in self._finger_joints:
            self.data.qpos[self.model.jnt_qposadr[jid]] = 0.0
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.ctrl[self._arm_acts] = home
        self.data.ctrl[self._finger_acts] = 0.0
        if self._weld_id is not None:
            self._set_eq_active(0)
        self._commanded_q = home.copy()
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------- state
    def q(self) -> np.ndarray:
        return np.asarray(self.data.qpos[:7], dtype=float).copy()

    def dq(self) -> np.ndarray:
        return np.asarray(self.data.qvel[:7], dtype=float).copy()

    def flange_position(self) -> np.ndarray:
        return np.asarray(self.data.site("flange").xpos, dtype=float).copy()

    def tcp_position(self) -> np.ndarray:
        return np.asarray(self.data.site("tcp").xpos, dtype=float).copy()

    def flange_pose(self) -> Pose:
        return Pose(
            tuple(float(v) for v in self.data.site("flange").xpos),
            matrix_to_quat_wxyz(self.data.site("flange").xmat.reshape(3, 3)),
        )

    def tcp_pose(self) -> Pose:
        return Pose(
            tuple(float(v) for v in self.data.site("tcp").xpos),
            matrix_to_quat_wxyz(self.data.site("tcp").xmat.reshape(3, 3)),
        )

    def commanded_tcp(self) -> Pose:
        """由持续指令关节角做 FK，再叠加工具偏移。"""

        ik = _firmware_ik()
        pos, R = ik.fk(self._commanded_q)
        tcp_pos = pos + R @ np.array([0.0, 0.0, self.cfg.tcp_offset_m])
        return Pose(tuple(float(v) for v in tcp_pos), matrix_to_quat_wxyz(R))

    def gripper_opening_m(self) -> float:
        q_right = float(self.data.qpos[self.model.jnt_qposadr[self._finger_joints[0]]])
        q_left = float(self.data.qpos[self.model.jnt_qposadr[self._finger_joints[1]]])
        closure = (q_right + q_left) / 2.0
        return max(0.0, 2.0 * (self.cfg.gripper.finger_inner_x_m - closure))

    def gripper_force_n(self) -> float:
        return float(np.sum(np.abs(self.data.actuator_force[self._finger_acts])) / 2.0)

    def gripper_state(self) -> GripperState:
        opening = self.gripper_opening_m()
        closure = 1.0 - opening / max(self.cfg.gripper.open_m, 1e-9)
        return GripperState(
            opening_m=opening,
            closure=max(0.0, min(1.0, closure)),
            force_n=self.gripper_force_n(),
        )

    def arm_state(self) -> ArmState:
        tau = np.asarray(self.data.actuator_force[self._arm_acts], dtype=float)
        return ArmState(
            q=tuple(float(v) for v in self.data.qpos[:7]),
            dq=tuple(float(v) for v in self.data.qvel[:7]),
            tau=tuple(float(v) for v in tau),
            tcp=self.tcp_pose(),
            flange=self.flange_pose(),
            mode="SIM",
            faulted=False,
        )

    def observe(self, *, frame_id: int, phase: str) -> Observation:
        head = self.cameras.render(
            self.cfg.cameras.head.name,
            self.cfg.cameras.head.width,
            self.cfg.cameras.head.height,
        )
        wrist = self.cameras.render(
            self.cfg.cameras.wrist.name,
            self.cfg.cameras.wrist.width,
            self.cfg.cameras.wrist.height,
        )
        return Observation(
            frame_id=frame_id,
            timestamp_s=time.monotonic(),
            images={"head": head.image, "wrist": wrist.image},
            image_age_s={"head": head.age_s, "wrist": wrist.age_s},
            arm=self.arm_state(),
            gripper=self.gripper_state(),
            phase=phase,
            commanded_tcp=self.commanded_tcp(),
        )

    # ------------------------------------------------------------------ block
    def block_position(self) -> np.ndarray:
        adr = int(self.model.jnt_qposadr[self._block_joint])
        return np.asarray(self.data.qpos[adr : adr + 3], dtype=float).copy()

    def set_block_position(self, xyz, quat=(1.0, 0.0, 0.0, 0.0)) -> None:
        adr = int(self.model.jnt_qposadr[self._block_joint])
        self.data.qpos[adr : adr + 3] = xyz
        self.data.qpos[adr + 3 : adr + 7] = quat
        dof = int(self.model.jnt_dofadr[self._block_joint])
        self.data.qvel[dof : dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    # --------------------------------------------------------------- stepping
    def _apply_gravity_compensation(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[:7] = self.data.qfrc_bias[:7]
        self.data.qfrc_applied[7:] = 0.0

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            self._apply_gravity_compensation()
            mujoco.mj_step(self.model, self.data)
            if self._viewer is not None:
                self._viewer_counter += 1
                if self._viewer_counter % self._sync_every == 0:
                    self._viewer.sync()
            if self._realtime:
                self._pace()

    def _pace(self) -> None:
        """按仿真时间 1x 播放，让 viewer 里的动作可观察。"""
        dt = float(self.model.opt.timestep)
        now = time.perf_counter()
        if self._next_tick < now - 0.1:
            self._next_tick = now
        self._next_tick += dt / self._speed
        delay = self._next_tick - now
        if delay > 0:
            time.sleep(min(delay, 0.02))

    def set_gripper(self, closure: float) -> None:
        closure = max(0.0, min(1.0, float(closure)))
        value = closure * 0.042726
        self.data.ctrl[self._finger_acts] = value

    def close_gripper(self) -> None:
        self.set_gripper(1.0)

    def open_gripper(self) -> None:
        self.set_gripper(0.0)

    def execute_joints(
        self,
        q_target,
        *,
        tcp_target: Pose | None = None,
        max_step_rad: float = 0.01,
        timeout_s: float | None = None,
        settle_s: float = 0.25,
        pos_tolerance_m: float | None = None,
    ) -> ExecutionResult:
        started = time.monotonic()
        q_target = np.asarray(q_target, dtype=float)
        if q_target.shape != (7,) or not np.all(np.isfinite(q_target)):
            return ExecutionResult(False, False, "invalid joint target", 0.0)
        timeout_s = timeout_s or self.cfg.step_timeout_s
        dt = float(self.model.opt.timestep)
        q_start = self.q()
        max_delta = float(np.max(np.abs(q_target - q_start)))
        ramp_steps = max(1, int(math.ceil(max_delta / max_step_rad)))
        limit_steps = max(1, int(timeout_s / dt))

        for k in range(ramp_steps):
            if k > limit_steps:
                return ExecutionResult(
                    False, False, "timeout", time.monotonic() - started, tuple(self.q())
                )
            alpha = (k + 1) / ramp_steps
            self.data.ctrl[self._arm_acts] = q_start + alpha * (q_target - q_start)
            self.step()
        for _ in range(max(1, int(settle_s / dt))):
            self.step()

        elapsed = time.monotonic() - started
        q_err = float(np.max(np.abs(self.q() - q_target)))
        if tcp_target is not None:
            pos_err = float(np.linalg.norm(self.tcp_position() - np.asarray(tcp_target.position)))
            from ..rotations import quat_angle

            rot_err = quat_angle(self.tcp_pose().quaternion_wxyz, tcp_target.quaternion_wxyz)
            pos_tol = 0.004 if pos_tolerance_m is None else float(pos_tolerance_m)
            ok = pos_err <= pos_tol and rot_err <= 0.03
            error = None if ok else f"tcp error {pos_err:.4f} m / {rot_err:.4f} rad"
        else:
            ok = q_err <= 0.02
            error = None if ok else f"joint error {q_err:.4f} rad"
        if ok:
            self._commanded_q = q_target.copy()
        return ExecutionResult(
            ok=ok,
            reached=ok,
            error=error,
            elapsed_s=elapsed,
            q=tuple(float(v) for v in self.q()),
            tcp=self.tcp_pose(),
        )

    # ---------------------------------------------------------------- grasping
    def _set_eq_active(self, value: int) -> None:
        target = self.data if hasattr(self.data, "eq_active") else self.model
        target.eq_active[self._weld_id] = value

    def set_grasp(self, enabled: bool) -> None:
        """焊接抓取，仅用于 weld 调试模式。"""
        if self._weld_id is None:
            raise RuntimeError("model has no grasp_weld equality")
        if not enabled:
            self._set_eq_active(0)
            return
        # 用当前相对位姿重写 relpose，避免激活时把方块拽到初始位置。
        gid = self._gripper_base
        rid = self._block_body
        p_g = np.asarray(self.data.xpos[gid], dtype=float)
        R_g = np.asarray(self.data.xmat[gid], dtype=float).reshape(3, 3)
        p_b = np.asarray(self.data.xpos[rid], dtype=float)
        R_b = np.asarray(self.data.xmat[rid], dtype=float).reshape(3, 3)
        rel_pos = R_g.T @ (p_b - p_g)
        rel_quat = matrix_to_quat_wxyz(R_g.T @ R_b)
        self.model.eq_data[self._weld_id, 3:6] = rel_pos
        self.model.eq_data[self._weld_id, 6:10] = rel_quat
        self._set_eq_active(1)

    def self_contacts(self) -> list[tuple[str, str]]:
        out = []
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            n1 = self.model.geom(con.geom1).name
            n2 = self.model.geom(con.geom2).name
            if n1.endswith("_col") or n2.endswith("_col"):
                out.append((n1, n2))
        return out


_FIRMWARE_IK = None


def _firmware_ik():
    global _FIRMWARE_IK
    if _FIRMWARE_IK is None:
        from .ik import FirmwareIK

        _FIRMWARE_IK = FirmwareIK.from_source()
    return _FIRMWARE_IK
