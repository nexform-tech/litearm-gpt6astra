"""不可变数据模型。命令状态与实测状态分开保存，互不覆盖。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .rotations import quat_wxyz_to_matrix


@dataclass(frozen=True)
class Pose:
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        position = tuple(float(v) for v in self.position)
        quat = tuple(float(v) for v in self.quaternion_wxyz)
        if len(position) != 3 or not all(math.isfinite(v) for v in position):
            raise ValueError("position must be 3 finite numbers")
        if len(quat) != 4 or not all(math.isfinite(v) for v in quat):
            raise ValueError("quaternion_wxyz must be 4 finite numbers")
        norm = math.sqrt(sum(v * v for v in quat))
        if abs(norm - 1.0) > 0.02:
            raise ValueError("quaternion_wxyz must be unit length")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "quaternion_wxyz", quat)

    def matrix(self) -> np.ndarray:
        return quat_wxyz_to_matrix(self.quaternion_wxyz)

    def as_dict(self) -> dict[str, Any]:
        return {
            "position": list(self.position),
            "quaternion_wxyz": list(self.quaternion_wxyz),
        }


@dataclass(frozen=True)
class ArmState:
    q: tuple[float, ...]
    dq: tuple[float, ...]
    tau: tuple[float, ...]
    tcp: Pose
    flange: Pose
    mode: str = "SIM"
    faulted: bool = False


@dataclass(frozen=True)
class GripperState:
    opening_m: float
    closure: float
    force_n: float = 0.0
    faulted: bool = False

    @property
    def opening_mm(self) -> float:
        return self.opening_m * 1000.0


@dataclass(frozen=True)
class Observation:
    frame_id: int
    timestamp_s: float
    images: dict[str, np.ndarray]
    image_age_s: dict[str, float]
    arm: ArmState
    gripper: GripperState
    phase: str
    commanded_tcp: Pose


@dataclass(frozen=True)
class Decision:
    status: str
    tcp_target: Pose
    gripper: int
    confidence: float
    evidence: dict[str, bool]
    reason: str
    clamped: bool = False
    proposed_translation_m: float = 0.0
    proposed_rotation_rad: float = 0.0


@dataclass(frozen=True)
class ExecutionResult:
    ok: bool
    reached: bool
    error: str | None
    elapsed_s: float
    q: tuple[float, ...] = ()
    tcp: Pose | None = None


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    score: float
    reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StepRecord:
    """steps.jsonl 的一行；proposal 与 executed 必须分开。"""

    step: int
    frame_id: int
    phase_before: str
    phase_after: str
    proposal: dict[str, Any] | None
    decision: dict[str, Any] | None
    executed: dict[str, Any]
    ik: dict[str, Any] | None = None
    images: dict[str, str] = field(default_factory=dict)
    verification: dict[str, Any] | None = None


@dataclass(frozen=True)
class RolloutResult:
    success: bool
    score: float
    steps: int
    phase: str
    verification: VerificationResult
    errors: tuple[str, ...] = ()
    elapsed_s: float = 0.0
    seed: int = 0
    mode: str = "scripted"
