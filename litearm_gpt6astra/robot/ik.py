"""固件 kin.c/ik.c 的 ctypes 绑定。

所有数组按 float32 传给固件，与 MCU 上的单精度一致。IK 返回 ``(q, ok)``；
``ok=False`` 时 q 是迭代最优解，调用方不得把它当作可达。
"""

from __future__ import annotations

import ctypes
import sys
from pathlib import Path

import numpy as np

from ..rotations import matrix_to_rpy, rpy_to_matrix

N = 7
Q_MIN = np.array(
    [-2.809547, -1.727547, -2.809547, -3.071547, -2.809547, -1.553547, -1.553547],
    dtype=np.float32,
)
Q_MAX = np.array(
    [2.809547, 1.727547, 2.809547, 0.017547, 2.809547, 1.553547, 1.553547],
    dtype=np.float32,
)


def _floats(values, length: int, label: str):
    v = np.asarray(values, dtype=np.float32).reshape(-1)
    if v.size != length:
        raise ValueError(f"{label}: expected {length} values, got {v.size}")
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{label}: non-finite value")
    return (ctypes.c_float * length)(*v.tolist())


def default_library_path() -> Path:
    return Path(__file__).resolve().parents[1] / "_native" / "libkinik.so"


class FirmwareIK:
    def __init__(self, library_path: str | Path):
        path = Path(library_path)
        if not path.is_file():
            raise FileNotFoundError(f"firmware IK library not found: {path}")
        self.path = path
        self.lib = ctypes.CDLL(str(path))
        self.lib.kin_init.argtypes = []
        self.lib.kin_init.restype = None
        self.lib.kin_fk.argtypes = [ctypes.POINTER(ctypes.c_float)] * 3
        self.lib.kin_fk.restype = None
        self.lib.ik_solve.argtypes = [ctypes.POINTER(ctypes.c_float)] * 6
        self.lib.ik_solve.restype = ctypes.c_bool
        self.lib.kin_jacobian.argtypes = [ctypes.POINTER(ctypes.c_float)] * 2
        self.lib.kin_jacobian.restype = None
        self.lib.kin_init()

    @classmethod
    def from_source(
        cls,
        firmware_root: str | Path | None = None,
        *,
        build: bool = True,
        library_path: str | Path | None = None,
    ) -> FirmwareIK:
        target = Path(library_path) if library_path else default_library_path()

        def _build() -> Path:
            scripts = Path(__file__).resolve().parents[2] / "scripts"
            if str(scripts) not in sys.path:
                sys.path.insert(0, str(scripts))
            from build_firmware_ik import build as build_lib

            return build_lib(firmware_root, target)

        if not target.is_file():
            if not build:
                raise FileNotFoundError(
                    f"{target} missing; run `python scripts/build_firmware_ik.py`"
                )
            target = _build()
        try:
            return cls(target)
        except OSError:
            # 预编译库与当前平台不匹配时，尝试用 vendored 源码重建。
            if not build:
                raise
            return cls(_build())

    def fk(self, q) -> tuple[np.ndarray, np.ndarray]:
        qa = _floats(q, N, "q")
        pos = (ctypes.c_float * 3)()
        rot = (ctypes.c_float * 9)()
        self.lib.kin_fk(qa, pos, rot)
        return (
            np.array(pos[:], dtype=float),
            np.array(rot[:], dtype=float).reshape(3, 3),
        )

    def fk_rpy(self, q) -> tuple[np.ndarray, tuple[float, float, float]]:
        pos, rot = self.fk(q)
        return pos, matrix_to_rpy(rot)

    def solve(self, pos, rotation, seed, q_min=Q_MIN, q_max=Q_MAX):
        out = (ctypes.c_float * N)()
        ok = self.lib.ik_solve(
            _floats(pos, 3, "pos"),
            _floats(np.asarray(rotation, dtype=float).reshape(-1), 9, "rotation"),
            _floats(seed, N, "seed"),
            _floats(q_min, N, "q_min"),
            _floats(q_max, N, "q_max"),
            out,
        )
        return np.array(out[:], dtype=float), bool(ok)

    def jacobian(self, q) -> np.ndarray:
        J = (ctypes.c_float * (6 * N))()
        self.lib.kin_jacobian(_floats(q, N, "q"), J)
        return np.array(J[:], dtype=float).reshape(6, N)

    def solve_local(
        self,
        pos,
        rotation,
        seed,
        q_min=Q_MIN,
        q_max=Q_MAX,
        *,
        pos_tol: float = 5e-4,
        rot_tol: float = 5e-3,
        max_iter: int = 400,
        step_clamp: float = 0.12,
        damping: float = 1e-3,
        max_joint_jump: float | None = None,
    ):
        """从 seed 出发的局部 DLS，避免固件随机重启跳到另一构型。"""
        from ..rotations import rotation_log

        pos_d = np.asarray(pos, dtype=float).reshape(3)
        R_d = np.asarray(rotation, dtype=float).reshape(3, 3)
        q = np.asarray(seed, dtype=float).reshape(N).copy()
        q_min = np.asarray(q_min, dtype=float)
        q_max = np.asarray(q_max, dtype=float)
        lam = float(damping)
        best = (q.copy(), float("inf"), float("inf"))
        for _ in range(max_iter):
            p, R = self.fk(q)
            e_pos = pos_d - p
            e_rot = rotation_log(R_d @ R.T)
            pe = float(np.linalg.norm(e_pos))
            re = float(np.linalg.norm(e_rot))
            if pe < best[1] and re < 0.1:
                best = (q.copy(), pe, re)
            if pe < pos_tol and re < rot_tol:
                if max_joint_jump is not None and np.max(np.abs(q - seed)) > max_joint_jump:
                    return q, False
                return q, True
            J = self.jacobian(q)
            e = np.concatenate([e_pos, e_rot])
            A = J @ J.T + lam * np.eye(6)
            try:
                dq = J.T @ np.linalg.solve(A, e)
            except np.linalg.LinAlgError:
                lam *= 4.0
                continue
            max_dq = float(np.max(np.abs(dq)))
            if max_dq > step_clamp:
                dq *= step_clamp / max_dq
            q_new = np.clip(q + dq, q_min, q_max)
            p_new, R_new = self.fk(q_new)
            new_err = float(
                np.linalg.norm(pos_d - p_new) + np.linalg.norm(rotation_log(R_d @ R_new.T))
            )
            old_err = pe + re
            if new_err < old_err:
                lam = max(1e-6, lam * 0.5)
                q = q_new
            else:
                lam = min(1.0, lam * 2.0)
        q, pe, re = best
        ok = pe < pos_tol and re < rot_tol
        if max_joint_jump is not None and np.max(np.abs(q - seed)) > max_joint_jump:
            ok = False
        return q, ok

    def solve_pose(self, pose, seed, q_min=Q_MIN, q_max=Q_MAX):
        pose = np.asarray(pose, dtype=float).reshape(-1)
        if pose.size != 6:
            raise ValueError("pose must be [x, y, z, roll, pitch, yaw]")
        return self.solve(pose[:3], rpy_to_matrix(pose[3:]), seed, q_min, q_max)
