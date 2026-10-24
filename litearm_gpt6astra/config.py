"""YAML 配置加载与严格校验。

未知键、错误类型、非有限值一律报错，避免拼写错误静默生效。
"""

from __future__ import annotations

import math
import types
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml


@dataclass(frozen=True)
class SceneConfig:
    model_path: Path
    timestep: float
    table_z: float
    block_size_m: float
    start_x: tuple[float, float]
    start_y: tuple[float, float]
    target_center: tuple[float, float]
    target_radius_m: float


@dataclass(frozen=True)
class CameraConfig:
    name: str
    width: int
    height: int


@dataclass(frozen=True)
class CamerasConfig:
    head: CameraConfig
    wrist: CameraConfig
    max_age_s: float


@dataclass(frozen=True)
class SafetyConfig:
    workspace_x: tuple[float, float]
    workspace_y: tuple[float, float]
    workspace_z: tuple[float, float]
    joint_margin_rad: float
    max_step_m: float
    max_step_near_grasp_m: float
    max_rotation_rad: float
    max_rotation_near_grasp_rad: float
    ik_pos_tol_m: float
    ik_rot_tol_rad: float
    max_joint_jump_rad: float
    gross_jump_factor: float


@dataclass(frozen=True)
class GripperConfig:
    mode: str
    open_m: float
    closed_m: float
    finger_inner_x_m: float
    friction: float
    close_time_s: float
    trial_lift_m: float


@dataclass(frozen=True)
class AstraConfig:
    api_style: str
    model: str
    base_url: str
    api_key_env: str
    proxy: str
    timeout_s: float
    confidence_min: float


@dataclass(frozen=True)
class AppConfig:
    seed: int
    home_q: tuple[float, ...]
    tcp_offset_m: float
    max_steps: int
    step_timeout_s: float
    scene: SceneConfig
    cameras: CamerasConfig
    safety: SafetyConfig
    gripper: GripperConfig
    astra: AstraConfig

    def __post_init__(self) -> None:
        if len(self.home_q) != 7:
            raise ValueError("home_q must have 7 values")
        if self.tcp_offset_m <= 0:
            raise ValueError("tcp_offset_m must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if self.step_timeout_s <= 0:
            raise ValueError("step_timeout_s must be positive")


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _build(tp: Any, value: Any, path: str) -> Any:
    origin = get_origin(tp)
    args = get_args(tp)

    if is_dataclass(tp):
        if not isinstance(value, dict):
            raise ValueError(f"{path}: expected mapping")
        hints = get_type_hints(tp)
        known = {f.name for f in fields(tp)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        missing = known - set(value)
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        kwargs = {k: _build(hints[k], value[k], f"{path}.{k}") for k in known}
        result = tp(**kwargs)
        return result

    if tp is bool:
        if type(value) is not bool:
            raise ValueError(f"{path}: expected bool")
        return value
    if tp is int:
        if type(value) is not int:
            raise ValueError(f"{path}: expected int")
        return value
    if tp is float:
        if not _is_number(value):
            raise ValueError(f"{path}: expected finite number")
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected string")
        return value
    if tp is Path:
        if not isinstance(value, str):
            raise ValueError(f"{path}: expected path string")
        return Path(value)

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path}: expected list")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_build(args[0], v, f"{path}[{i}]") for i, v in enumerate(value))
        if len(value) != len(args):
            raise ValueError(f"{path}: expected {len(args)} values")
        return tuple(
            _build(arg, v, f"{path}[{i}]")
            for i, (arg, v) in enumerate(zip(args, value, strict=True))
        )

    if origin in (types.UnionType,):
        # 本配置不使用 Optional；保留明确错误。
        raise ValueError(f"{path}: unsupported union type")

    raise ValueError(f"{path}: unsupported type {tp!r}")


def _validate_ranges(cfg: AppConfig) -> None:
    safety = cfg.safety
    for name in ("workspace_x", "workspace_y", "workspace_z"):
        lo, hi = getattr(safety, name)
        if lo >= hi:
            raise ValueError(f"safety.{name}: low must be < high")
    for name in ("start_x", "start_y"):
        lo, hi = getattr(cfg.scene, name)
        if lo >= hi:
            raise ValueError(f"scene.{name}: low must be < high")
    if safety.joint_margin_rad < 0:
        raise ValueError("safety.joint_margin_rad must be >= 0")
    for name in (
        "max_step_m",
        "max_step_near_grasp_m",
        "max_rotation_rad",
        "max_rotation_near_grasp_rad",
        "ik_pos_tol_m",
        "ik_rot_tol_rad",
        "max_joint_jump_rad",
    ):
        if getattr(safety, name) <= 0:
            raise ValueError(f"safety.{name} must be positive")
    if safety.gross_jump_factor <= 1:
        raise ValueError("safety.gross_jump_factor must be > 1")
    if cfg.gripper.mode not in ("contact", "weld"):
        raise ValueError("gripper.mode must be contact or weld")
    if cfg.gripper.open_m <= cfg.gripper.closed_m:
        raise ValueError("gripper.open_m must be > gripper.closed_m")
    if cfg.cameras.max_age_s <= 0:
        raise ValueError("cameras.max_age_s must be positive")
    if cfg.scene.timestep <= 0:
        raise ValueError("scene.timestep must be positive")
    if cfg.scene.block_size_m <= 0:
        raise ValueError("scene.block_size_m must be positive")
    if cfg.scene.target_radius_m <= 0:
        raise ValueError("scene.target_radius_m must be positive")
    for cam in (cfg.cameras.head, cfg.cameras.wrist):
        if cam.width <= 0 or cam.height <= 0:
            raise ValueError("camera width/height must be positive")
    if not 0.0 < cfg.astra.confidence_min <= 1.0:
        raise ValueError("astra.confidence_min must be in (0, 1]")
    if cfg.astra.api_style not in ("chat", "responses"):
        raise ValueError("astra.api_style must be 'chat' or 'responses'")


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    cfg = _build(AppConfig, raw, "config")
    _validate_ranges(cfg)
    if not cfg.scene.model_path.is_absolute():
        repo_root = config_path.parent.parent
        cfg = replace(
            cfg,
            scene=replace(cfg.scene, model_path=(repo_root / cfg.scene.model_path).resolve()),
        )
    return cfg
