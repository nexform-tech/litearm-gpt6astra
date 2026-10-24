"""执行前的确定性安全检查。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..config import SafetyConfig
from ..rotations import quat_angle, quat_slerp
from ..types import Pose
from .ik import Q_MAX, Q_MIN


class SafetyError(RuntimeError):
    """安全校验失败；调用方应锁存 held，而不是重试。"""


@dataclass
class StepResult:
    target: Pose
    translation_m: float
    rotation_rad: float
    clamped: bool


class SafetyChecker:
    def __init__(
        self,
        config: SafetyConfig,
        *,
        q_min=Q_MIN,
        q_max=Q_MAX,
        grasp_height_m: float = 0.030,
    ):
        self.cfg = config
        self.q_min = np.asarray(q_min, dtype=float)
        self.q_max = np.asarray(q_max, dtype=float)
        self.grasp_height_m = grasp_height_m
        if self.q_min.shape != (7,) or self.q_max.shape != (7,):
            raise ValueError("q_min/q_max must have 7 values")
        if np.any(self.q_min >= self.q_max):
            raise ValueError("q_min must be < q_max")

    # ------------------------------------------------------------------ basics
    def check_position(self, position) -> None:
        p = np.asarray(position, dtype=float).reshape(3)
        if not np.all(np.isfinite(p)):
            raise SafetyError("workspace: non-finite position")
        limits = (self.cfg.workspace_x, self.cfg.workspace_y, self.cfg.workspace_z)
        for i, (lo, hi) in enumerate(limits):
            if p[i] < lo or p[i] > hi:
                raise SafetyError(f"workspace: axis {i}={p[i]:.4f} outside [{lo}, {hi}]")

    def check_joints(self, q) -> None:
        q = np.asarray(q, dtype=float).reshape(7)
        if not np.all(np.isfinite(q)):
            raise SafetyError("joint: non-finite value")
        margin = self.cfg.joint_margin_rad
        lo = self.q_min + margin
        hi = self.q_max - margin
        bad = np.where((q < lo) | (q > hi))[0]
        if bad.size:
            i = int(bad[0])
            raise SafetyError(f"joint: j{i + 1}={q[i]:.4f} outside [{lo[i]:.4f}, {hi[i]:.4f}]")

    # --------------------------------------------------------------- step limits
    def step_limits(self, phase: str, near_grasp: bool) -> tuple[float, float]:
        if phase == "transfer":
            return self.cfg.max_step_m, self.cfg.max_rotation_rad
        if near_grasp:
            return self.cfg.max_step_near_grasp_m, self.cfg.max_rotation_near_grasp_rad
        return self.cfg.max_step_m, self.cfg.max_rotation_rad

    def clamp_step(
        self,
        current: Pose,
        target: Pose,
        *,
        phase: str,
        near_grasp: bool,
    ) -> StepResult:
        self.check_position(current.position)
        self.check_position(target.position)
        max_t, max_r = self.step_limits(phase, near_grasp)

        delta = np.asarray(target.position) - np.asarray(current.position)
        dist = float(np.linalg.norm(delta))
        angle = quat_angle(current.quaternion_wxyz, target.quaternion_wxyz)

        factor = self.cfg.gross_jump_factor
        if dist > factor * max_t + 1e-9:
            raise SafetyError(f"gross translation jump {dist:.4f} m rejected")
        if angle > factor * max_r + 1e-9:
            raise SafetyError(f"gross rotation jump {angle:.4f} rad rejected")

        clamped = False
        new_pos = np.asarray(target.position, dtype=float)
        new_quat = target.quaternion_wxyz
        if dist > max_t:
            new_pos = np.asarray(current.position) + delta / dist * max_t
            clamped = True
        if angle > max_r:
            new_quat = quat_slerp(current.quaternion_wxyz, target.quaternion_wxyz, max_r / angle)
            clamped = True
        return StepResult(
            target=Pose(tuple(float(v) for v in new_pos), new_quat),
            translation_m=dist,
            rotation_rad=angle,
            clamped=clamped,
        )

    # --------------------------------------------------------------- IK checks
    def check_ik(
        self,
        q_seed,
        q_solution,
        *,
        pos_error_m: float,
        rot_error_rad: float,
        max_joint_jump_rad: float | None = None,
    ) -> None:
        if pos_error_m > self.cfg.ik_pos_tol_m:
            raise SafetyError(f"IK position residual {pos_error_m:.5f} m too large")
        if rot_error_rad > self.cfg.ik_rot_tol_rad:
            raise SafetyError(f"IK rotation residual {rot_error_rad:.5f} rad too large")
        jump = float(np.max(np.abs(np.asarray(q_solution) - np.asarray(q_seed))))
        limit = self.cfg.max_joint_jump_rad if max_joint_jump_rad is None else max_joint_jump_rad
        if jump > limit + 1e-9:
            raise SafetyError(f"IK joint jump {jump:.4f} rad rejected")
        self.check_joints(q_solution)

    def check_path(self, start: Pose, end: Pose, *, samples: int = 12) -> None:
        a = np.asarray(start.position, dtype=float)
        b = np.asarray(end.position, dtype=float)
        for t in np.linspace(0.0, 1.0, samples):
            p = a + (b - a) * t
            self.check_position(p)
            if p[2] < self.cfg.workspace_z[0] + 0.002:
                raise SafetyError(f"path: TCP too low ({p[2]:.4f} m)")

    def check_self_contacts(self, contacts, *, enabled: bool = False) -> None:
        if not enabled:
            return
        links = {name for pair in contacts for name in pair if name.endswith("_col")}
        if len(links) >= 2:
            raise SafetyError(f"self collision: {sorted(links)}")


def quat_distance(a, b) -> float:
    return quat_angle(a, b)


def finite_pose(pose: Pose) -> bool:
    return all(math.isfinite(v) for v in pose.position) and all(
        math.isfinite(v) for v in pose.quaternion_wxyz
    )
