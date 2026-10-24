import math

import numpy as np
import pytest

from litearm_gpt6astra.rotations import (
    matrix_to_quat_wxyz,
    matrix_to_rpy,
    quat_angle,
    quat_multiply,
    quat_slerp,
    quat_wxyz_to_matrix,
    rpy_to_matrix,
)


def test_identity_roundtrip():
    q = (1.0, 0.0, 0.0, 0.0)
    R = quat_wxyz_to_matrix(q)
    assert np.allclose(R, np.eye(3), atol=1e-12)
    assert np.allclose(matrix_to_quat_wxyz(R), q, atol=1e-12)


def test_yaw_90_rpy_matches_quat():
    R = rpy_to_matrix((0.0, 0.0, math.pi / 2))
    q = matrix_to_quat_wxyz(R)
    assert np.allclose(quat_wxyz_to_matrix(q), R, atol=1e-12)
    assert np.allclose(matrix_to_rpy(R), (0.0, 0.0, math.pi / 2), atol=1e-9)


def test_tool_down_quaternion():
    q = (0.0, 1.0, 0.0, 0.0)
    R = quat_wxyz_to_matrix(q)
    assert np.allclose(R @ np.array([0.0, 0.0, 1.0]), [0.0, 0.0, -1.0], atol=1e-12)


def test_quat_multiply_and_angle():
    a = matrix_to_quat_wxyz(rpy_to_matrix((0.1, 0.2, 0.3)))
    b = matrix_to_quat_wxyz(rpy_to_matrix((-0.2, 0.05, 0.4)))
    c = quat_multiply(a, b)
    R = quat_wxyz_to_matrix(a) @ quat_wxyz_to_matrix(b)
    assert np.allclose(quat_wxyz_to_matrix(c), R, atol=1e-12)
    assert quat_angle(c, c) == 0.0


def test_matrix_normalizes_non_unit_quat():
    q = (2.0, 0.0, 0.0, 0.0)
    assert np.allclose(quat_wxyz_to_matrix(q), np.eye(3), atol=1e-12)


def test_gimbal_lock_keeps_matrix():
    R = rpy_to_matrix((0.3, math.pi / 2, 0.7))
    rpy = matrix_to_rpy(R)
    assert abs(rpy[1] - math.pi / 2) < 1e-9
    assert np.allclose(rpy_to_matrix(rpy), R, atol=1e-6)


def test_slerp_endpoints():
    a = matrix_to_quat_wxyz(rpy_to_matrix((0.0, 0.0, 0.0)))
    b = matrix_to_quat_wxyz(rpy_to_matrix((0.0, 0.0, 1.0)))
    assert np.allclose(quat_slerp(a, b, 0.0), a, atol=1e-12)
    assert np.allclose(quat_slerp(a, b, 1.0), b, atol=1e-12)


def test_zero_quaternion_rejected():
    with pytest.raises(ValueError):
        quat_wxyz_to_matrix((0.0, 0.0, 0.0, 0.0))
