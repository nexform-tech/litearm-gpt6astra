"""四元数(wxyz)、旋转矩阵、固定轴 ZYX RPY 的最小工具。

坐标系约定：向量列乘，``v_world = R @ v_local``。RPY 与固件
``kin_rpy_to_rot`` 一致：``R = Rz(yaw) @ Ry(pitch) @ Rx(roll)``。
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Quat = tuple[float, float, float, float]


def _normalize(q: Sequence[float]) -> Quat:
    v = np.asarray(q, dtype=float)
    if v.shape != (4,):
        raise ValueError("quaternion must have 4 components")
    n = float(np.linalg.norm(v))
    if not math.isfinite(n) or n < 1e-12:
        raise ValueError("quaternion is zero or non-finite")
    return (float(v[0] / n), float(v[1] / n), float(v[2] / n), float(v[3] / n))


def quat_wxyz_to_matrix(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = _normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def matrix_to_quat_wxyz(R: Sequence[Sequence[float]]) -> Quat:
    m = np.asarray(R, dtype=float)
    if m.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    trace = float(np.trace(m))
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return _normalize((w, x, y, z))


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def matrix_to_rpy(R: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    m = np.asarray(R, dtype=float)
    if m.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    sy = max(-1.0, min(1.0, -float(m[2, 0])))
    pitch = math.asin(sy)
    if abs(math.cos(pitch)) > 1e-8:
        roll = math.atan2(m[2, 1], m[2, 2])
        yaw = math.atan2(m[1, 0], m[0, 0])
    else:
        # 万向锁：yaw 不可观测，按固件约定强制 yaw=0，用 roll 吸收。
        roll = math.atan2(-m[1, 2], m[1, 1])
        yaw = 0.0
    return float(roll), float(pitch), float(yaw)


def quat_multiply(a: Sequence[float], b: Sequence[float]) -> Quat:
    aw, ax, ay, az = _normalize(a)
    bw, bx, by, bz = _normalize(b)
    return _normalize(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        )
    )


def quat_conjugate(q: Sequence[float]) -> Quat:
    w, x, y, z = _normalize(q)
    return (w, -x, -y, -z)


def quat_angle(a: Sequence[float], b: Sequence[float]) -> float:
    dot = abs(sum(x * y for x, y in zip(_normalize(a), _normalize(b), strict=True)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def quat_slerp(a: Sequence[float], b: Sequence[float], t: float) -> Quat:
    qa = np.asarray(_normalize(a), dtype=float)
    qb = np.asarray(_normalize(b), dtype=float)
    dot = float(np.dot(qa, qb))
    if dot < 0:
        qb = -qb
        dot = -dot
    if dot > 0.9995:
        return _normalize(qa + t * (qb - qa))
    theta = math.acos(max(-1.0, min(1.0, dot)))
    s = math.sin(theta)
    out = math.sin((1 - t) * theta) / s * qa + math.sin(t * theta) / s * qb
    return _normalize(out)


def rotvec_angle(a: Sequence[float], b: Sequence[float]) -> float:
    """两个四元数之间的最短旋转角（rad）。"""
    return quat_angle(a, b)


def rotation_log(R: Sequence[Sequence[float]]) -> np.ndarray:
    """SO(3) log：旋转矩阵 -> 旋转向量（轴×角）。"""
    m = np.asarray(R, dtype=float)
    if m.shape != (3, 3):
        raise ValueError("rotation must be 3x3")
    cos_theta = max(-1.0, min(1.0, (float(np.trace(m)) - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    if theta < 1e-8:
        return np.zeros(3, dtype=float)
    if abs(math.pi - theta) < 1e-6:
        # theta≈pi：从 R+I 的列取轴
        A = (m + np.eye(3)) / 2.0
        axis = A[:, int(np.argmax(np.diag(A)))]
        axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
        return axis * theta
    vee = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]])
    return vee * (theta / (2.0 * math.sin(theta)))
