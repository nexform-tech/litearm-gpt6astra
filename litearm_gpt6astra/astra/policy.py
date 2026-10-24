"""proposal 校验、阶段门控与动作锁存。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..robot.safety import SafetyChecker, SafetyError
from ..rotations import quat_angle
from ..types import Decision, Pose
from .client import Halt
from .schema import EVIDENCE_FIELDS, REQUIRED_FIELDS, STATUSES

ALLOWED = {
    "approach": {"move", "grasp", "hold", "stop"},
    "pregrasp": {"move", "grasp", "hold", "stop"},
    "transfer": {"move", "release", "hold", "stop"},
    "released": {"move", "done", "hold", "stop"},
    "done": {"hold", "stop"},
    "held": {"hold", "stop"},
}


def _finite_vector(value, length: int, label: str) -> np.ndarray:
    if not isinstance(value, list) or len(value) != length:
        raise Halt(f"{label}: expected {length} numbers")
    if any(type(v) not in (int, float) for v in value):
        raise Halt(f"{label}: non-numeric value")
    out = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(out)):
        raise Halt(f"{label}: non-finite value")
    return out


def _pose(raw: dict, label: str) -> Pose:
    if set(raw) != {"position", "quaternion_wxyz"}:
        raise Halt(f"{label}: invalid fields")
    position = _finite_vector(raw["position"], 3, f"{label}.position")
    quat = _finite_vector(raw["quaternion_wxyz"], 4, f"{label}.quaternion_wxyz")
    norm = float(np.linalg.norm(quat))
    if abs(norm - 1.0) > 0.02:
        raise Halt(f"{label}: quaternion is not unit length")
    return Pose(tuple(float(v) for v in position), tuple(float(v / norm) for v in quat))


@dataclass
class PhaseState:
    phase: str = "approach"

    def advance(self, decision: Decision, *, executed_ok: bool) -> None:
        if decision.status == "hold" or decision.status == "stop":
            return
        if decision.status == "grasp":
            if self.phase == "approach":
                self.phase = "pregrasp"
            elif self.phase == "pregrasp" and executed_ok:
                self.phase = "transfer"
        elif decision.status == "release" and self.phase == "transfer" and executed_ok:
            self.phase = "released"
        elif decision.status == "done" and self.phase == "released" and executed_ok:
            self.phase = "done"

    def set_held(self) -> None:
        self.phase = "held"


class PolicyValidator:
    def __init__(
        self,
        safety: SafetyChecker,
        *,
        confidence_min: float = 0.80,
    ):
        self.safety = safety
        self.confidence_min = float(confidence_min)
        self.last_frame_id = -1

    def validate(
        self,
        raw: dict,
        *,
        current: Pose,
        phase: str,
        frame_id: int,
        gripper_closed: bool,
        block_position=None,
        target_center=None,
        target_z: float = 0.0,
    ) -> Decision:
        if not isinstance(raw, dict) or set(raw) != set(REQUIRED_FIELDS):
            raise Halt("invalid proposal fields; joints/trajectories/extra fields forbidden")
        if raw["status"] not in STATUSES:
            raise Halt("unknown status")
        if not isinstance(raw["reason"], str) or not raw["reason"].strip():
            raise Halt("missing reason")
        target = _pose(raw["tcp_target"], "tcp_target")
        if type(raw["gripper"]) is not int or raw["gripper"] not in (0, 1):
            raise Halt("gripper must be 0 or 1")
        confidence = raw["confidence"]
        if type(confidence) not in (int, float) or not math.isfinite(float(confidence)):
            raise Halt("confidence is not finite")
        if not self.confidence_min <= float(confidence) <= 1.0:
            raise Halt(f"confidence below {self.confidence_min}")
        for key in EVIDENCE_FIELDS:
            if type(raw[key]) is not bool:
                raise Halt(f"{key} must be boolean")
        if raw["status"] == "stop":
            raise Halt("Astra STOP: " + raw["reason"])
        if phase not in ALLOWED:
            raise Halt(f"unknown phase {phase}")
        if raw["status"] not in ALLOWED[phase]:
            raise Halt(f"action {raw['status']} not allowed in phase {phase}")
        if frame_id <= self.last_frame_id:
            raise Halt("reused observation cannot advance phase")

        try:
            self.safety.check_position(current.position)
            self.safety.check_position(target.position)
        except SafetyError as exc:
            raise Halt(str(exc)) from None

        evidence = {key: bool(raw[key]) for key in EVIDENCE_FIELDS}
        status = raw["status"]
        gripper = int(raw["gripper"])

        # hold 是“请求新观测”，允许在证据不完整时使用；它不能带运动。
        if status == "hold":
            if np.linalg.norm(np.asarray(target.position) - np.asarray(current.position)) > 1e-6:
                raise Halt("hold includes translation")
            if quat_angle(target.quaternion_wxyz, current.quaternion_wxyz) > 1e-5:
                raise Halt("hold includes rotation")
            if gripper != int(gripper_closed):
                raise Halt("hold must match the current gripper state")
            self.last_frame_id = frame_id
            return Decision(
                status="hold",
                tcp_target=target,
                gripper=gripper,
                confidence=float(confidence),
                evidence=evidence,
                reason=raw["reason"],
                clamped=False,
            )

        if raw["uncertain"] or not (
            raw["target_visible"] and raw["hand_visible"] and raw["path_clear"]
        ):
            raise Halt("uncertainty, target/hand loss or unclear path: " + raw["reason"])

        if status == "move":
            if phase in ("approach", "pregrasp") and gripper != 0:
                raise Halt("approach move must keep the gripper open")
            if phase == "transfer" and gripper != 1:
                raise Halt("transfer move must keep the gripper closed")
            if phase == "released" and gripper != 0:
                raise Halt("released move must keep the gripper open")
            try:
                step = self.safety.clamp_step(
                    current,
                    target,
                    phase=phase,
                    near_grasp=evidence["near_grasp"],
                )
            except SafetyError as exc:
                raise Halt(str(exc)) from None
            target = step.target
            clamped = step.clamped
            trans = step.translation_m
            rot = step.rotation_rad
        else:
            clamped = False
            trans = 0.0
            rot = 0.0
            if status == "grasp":
                if gripper_closed:
                    raise Halt("grasp requires an open gripper")
                if gripper != 1:
                    raise Halt("grasp requires gripper=1")
                if not (evidence["object_between_fingers"] and evidence["tips_at_grasp_height"]):
                    raise Halt("grasp requires object_between_fingers and tips_at_grasp_height")
                if phase not in ("approach", "pregrasp"):
                    raise Halt(f"grasp not allowed in phase {phase}")
                if block_position is not None:
                    delta = np.asarray(block_position, dtype=float) - np.asarray(target.position)
                    if float(np.linalg.norm(delta[:2])) > 0.02:
                        raise Halt("grasp target is not over the block")
                    if abs(float(delta[2])) > 0.025:
                        raise Halt("grasp target is not at the block height")
            elif status == "release":
                if not gripper_closed:
                    raise Halt("release requires a closed gripper")
                if gripper != 0:
                    raise Halt("release requires gripper=0")
                if not evidence["object_in_target_zone"]:
                    raise Halt("release requires object_in_target_zone")
                if phase != "transfer":
                    raise Halt("release only allowed during transfer")
                if target_center is not None:
                    delta = np.asarray(target.position) - np.asarray(
                        [target_center[0], target_center[1], target_z]
                    )
                    if float(np.linalg.norm(delta[:2])) > 0.03:
                        raise Halt("release target is not over the target zone")
                    if float(abs(delta[2])) > 0.06:
                        raise Halt("release target is too high above the target zone")
            elif status in ("hold", "done", "stop"):
                if gripper != int(gripper_closed):
                    raise Halt("hold/done/stop must match the current gripper state")
            if np.linalg.norm(np.asarray(target.position) - np.asarray(current.position)) > 1e-6:
                raise Halt("stationary status includes translation")
            if quat_angle(target.quaternion_wxyz, current.quaternion_wxyz) > 1e-5:
                raise Halt("stationary status includes rotation")

        self.last_frame_id = frame_id
        return Decision(
            status=status,
            tcp_target=target,
            gripper=gripper,
            confidence=float(confidence),
            evidence=evidence,
            reason=raw["reason"],
            clamped=clamped,
            proposed_translation_m=trans,
            proposed_rotation_rad=rot,
        )
