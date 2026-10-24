import numpy as np
import pytest

from litearm_gpt6astra.astra.client import Halt
from litearm_gpt6astra.astra.policy import PhaseState, PolicyValidator
from litearm_gpt6astra.config import load_config
from litearm_gpt6astra.robot.safety import SafetyChecker
from litearm_gpt6astra.types import Pose

CUR = Pose((0.15, -0.15, 0.15), (0, 1, 0, 0))
DOWN = [0, 1, 0, 0]


def proposal(**kw):
    base = dict(
        status="move",
        tcp_target={"position": [0.15, -0.15, 0.145], "quaternion_wxyz": DOWN},
        gripper=0,
        confidence=0.9,
        target_visible=True,
        hand_visible=True,
        path_clear=True,
        near_grasp=True,
        object_between_fingers=False,
        tips_at_grasp_height=False,
        object_in_target_zone=False,
        uncertain=False,
        reason="ok",
    )
    base.update(kw)
    return base


@pytest.fixture
def validator():
    cfg = load_config("configs/sim.yaml")
    return PolicyValidator(SafetyChecker(cfg.safety), confidence_min=0.80)


def test_move_clamped_to_near_grasp_limit(validator):
    raw = proposal(tcp_target={"position": [0.15, -0.15, 0.143], "quaternion_wxyz": DOWN})
    d = validator.validate(raw, current=CUR, phase="approach", frame_id=1, gripper_closed=False)
    assert d.clamped
    moved = np.linalg.norm(np.array(d.tcp_target.position) - np.array(CUR.position))
    assert moved <= 0.005 + 1e-6


def test_gross_jump_rejected(validator):
    raw = proposal(tcp_target={"position": [0.15, -0.15, 0.05], "quaternion_wxyz": DOWN})
    with pytest.raises(Halt, match="gross"):
        validator.validate(raw, current=CUR, phase="approach", frame_id=1, gripper_closed=False)


def test_grasp_requires_evidence(validator):
    raw = proposal(
        status="grasp",
        gripper=1,
        near_grasp=True,
        tcp_target={"position": [0.15, -0.15, 0.15], "quaternion_wxyz": DOWN},
    )
    with pytest.raises(Halt, match="between"):
        validator.validate(raw, current=CUR, phase="approach", frame_id=1, gripper_closed=False)


def test_grasp_allowed_in_approach(validator):
    raw = proposal(
        status="grasp",
        gripper=1,
        near_grasp=True,
        object_between_fingers=True,
        tips_at_grasp_height=True,
        tcp_target={"position": [0.15, -0.15, 0.15], "quaternion_wxyz": DOWN},
    )
    d = validator.validate(
        raw,
        current=CUR,
        phase="approach",
        frame_id=1,
        gripper_closed=False,
        block_position=(0.15, -0.15, 0.135),
    )
    assert d.status == "grasp"


def test_grasp_requires_pregrasp_for_closure(validator):
    raw = proposal(
        status="grasp",
        gripper=1,
        near_grasp=True,
        object_between_fingers=True,
        tips_at_grasp_height=True,
        tcp_target={"position": [0.15, -0.15, 0.15], "quaternion_wxyz": DOWN},
    )
    d = validator.validate(
        raw,
        current=CUR,
        phase="pregrasp",
        frame_id=1,
        gripper_closed=False,
        block_position=(0.15, -0.15, 0.135),
    )
    assert d.status == "grasp"


def test_grasp_requires_open_gripper(validator):
    raw = proposal(
        status="grasp",
        gripper=1,
        near_grasp=True,
        object_between_fingers=True,
        tips_at_grasp_height=True,
        tcp_target={"position": [0.15, -0.15, 0.15], "quaternion_wxyz": DOWN},
    )
    with pytest.raises(Halt, match="open gripper"):
        validator.validate(
            raw,
            current=CUR,
            phase="pregrasp",
            frame_id=1,
            gripper_closed=True,
            block_position=(0.15, -0.15, 0.135),
        )


def test_hold_allowed_when_uncertain(validator):
    raw = proposal(
        status="hold",
        gripper=0,
        uncertain=True,
        target_visible=False,
        hand_visible=False,
        path_clear=False,
        tcp_target={"position": [0.15, -0.15, 0.15], "quaternion_wxyz": DOWN},
        reason="need a fresh wrist view",
    )
    d = validator.validate(raw, current=CUR, phase="approach", frame_id=1, gripper_closed=False)
    assert d.status == "hold"


def test_hold_with_motion_rejected(validator):
    raw = proposal(
        status="hold",
        gripper=0,
        uncertain=True,
        target_visible=False,
        tcp_target={"position": [0.16, -0.15, 0.15], "quaternion_wxyz": DOWN},
        reason="bad hold",
    )
    with pytest.raises(Halt, match="hold includes translation"):
        validator.validate(raw, current=CUR, phase="approach", frame_id=1, gripper_closed=False)


def test_old_frame_cannot_advance(validator):
    validator.validate(proposal(), current=CUR, phase="approach", frame_id=5, gripper_closed=False)
    with pytest.raises(Halt, match="reused"):
        validator.validate(
            proposal(), current=CUR, phase="approach", frame_id=5, gripper_closed=False
        )


def test_stop_halts(validator):
    with pytest.raises(Halt, match="STOP"):
        validator.validate(
            proposal(status="stop", reason="lost"),
            current=CUR,
            phase="approach",
            frame_id=1,
            gripper_closed=False,
        )


def test_transfer_move_requires_closed_gripper(validator):
    raw = proposal(
        gripper=0,
        near_grasp=False,
        tcp_target={"position": [0.15, -0.15, 0.13], "quaternion_wxyz": DOWN},
    )
    with pytest.raises(Halt, match="closed"):
        validator.validate(raw, current=CUR, phase="transfer", frame_id=1, gripper_closed=True)


def test_release_requires_target_zone(validator):
    current = Pose((0.20, -0.10, 0.05), (0, 1, 0, 0))
    raw = proposal(
        status="release",
        gripper=0,
        near_grasp=True,
        object_in_target_zone=True,
        tcp_target={"position": [0.20, -0.10, 0.05], "quaternion_wxyz": DOWN},
    )
    d = validator.validate(
        raw,
        current=current,
        phase="transfer",
        frame_id=1,
        gripper_closed=True,
        target_center=(0.20, -0.10),
    )
    assert d.status == "release"


def test_release_requires_closed_gripper(validator):
    current = Pose((0.20, -0.10, 0.05), (0, 1, 0, 0))
    raw = proposal(
        status="release",
        gripper=0,
        near_grasp=True,
        object_in_target_zone=True,
        tcp_target={"position": [0.20, -0.10, 0.05], "quaternion_wxyz": DOWN},
    )
    with pytest.raises(Halt, match="closed gripper"):
        validator.validate(
            raw,
            current=current,
            phase="transfer",
            frame_id=1,
            gripper_closed=False,
            target_center=(0.20, -0.10),
        )


def test_phase_state_transitions():
    p = PhaseState()
    assert p.phase == "approach"
    p.advance(_decision("grasp"), executed_ok=True)
    assert p.phase == "pregrasp"
    p.advance(_decision("grasp"), executed_ok=True)
    assert p.phase == "transfer"
    p.advance(_decision("release"), executed_ok=True)
    assert p.phase == "released"
    p.advance(_decision("done"), executed_ok=True)
    assert p.phase == "done"


def _decision(status):
    from litearm_gpt6astra.types import Decision

    return Decision(
        status=status,
        tcp_target=CUR,
        gripper=0,
        confidence=0.9,
        evidence={},
        reason="x",
    )
