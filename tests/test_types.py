import pytest

from litearm_gpt6astra.types import Decision, Pose


def test_pose_rejects_non_unit_quat():
    with pytest.raises(ValueError):
        Pose((0, 0, 0), (2, 0, 0, 0))


def test_pose_accepts_near_unit_quat():
    p = Pose((0, 0, 0), (1.0, 0.001, 0, 0))
    assert p.position == (0.0, 0.0, 0.0)


def test_decision_is_frozen():
    d = Decision(
        status="move",
        tcp_target=Pose((0, 0, 0), (1, 0, 0, 0)),
        gripper=0,
        confidence=0.9,
        evidence={},
        reason="x",
        clamped=False,
        proposed_translation_m=0.0,
        proposed_rotation_rad=0.0,
    )
    with pytest.raises(Exception):
        d.status = "stop"  # type: ignore[misc]
