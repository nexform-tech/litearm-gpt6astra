import numpy as np
import pytest

from litearm_gpt6astra.robot.ik import Q_MAX, Q_MIN, FirmwareIK


@pytest.fixture(scope="module")
def ik():
    return FirmwareIK.from_source()


def test_fk_ik_fk_roundtrip(ik):
    q = np.array([0.0, 0.6, 0.0, -1.2, 0.0, 0.7, 0.0])
    p, R = ik.fk(q)
    q2, ok = ik.solve(p, R, seed=np.zeros(7), q_min=Q_MIN, q_max=Q_MAX)
    assert ok
    p2, R2 = ik.fk(q2)
    assert np.linalg.norm(p2 - p) < 1e-3
    assert np.linalg.norm(R2 - R) < 1e-2


def test_zero_pose_is_tall(ik):
    p, R = ik.fk(np.zeros(7))
    assert abs(p[2] - 0.783) < 2e-3
    assert np.allclose(R, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-3)


def test_unreachable_returns_false(ik):
    _, ok = ik.solve(
        np.array([10.0, 10.0, 10.0]),
        np.eye(3),
        seed=np.zeros(7),
        q_min=Q_MIN,
        q_max=Q_MAX,
    )
    assert not ok


def test_wrong_length_raises(ik):
    with pytest.raises(ValueError):
        ik.fk([0.0] * 6)
