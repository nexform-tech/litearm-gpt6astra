import numpy as np
import pytest

from litearm_gpt6astra.config import load_config
from litearm_gpt6astra.robot.safety import SafetyChecker, SafetyError
from litearm_gpt6astra.types import Pose


@pytest.fixture
def checker():
    return SafetyChecker(load_config("configs/sim.yaml").safety)


def test_workspace_rejects_below_table(checker):
    with pytest.raises(SafetyError, match="workspace"):
        checker.check_position((0.1, 0.1, -0.1))


def test_joint_limits_with_margin(checker):
    q = np.array([0, 0.6, 0, -1.2, 0, 0.7, 0.0])
    checker.check_joints(q)
    q[3] = -3.5
    with pytest.raises(SafetyError, match="joint"):
        checker.check_joints(q)


def test_step_clamps_small_overshoot(checker):
    current = Pose((0.15, -0.15, 0.15), (0, 1, 0, 0))
    target = Pose((0.156, -0.15, 0.15), (0, 1, 0, 0))
    step = checker.clamp_step(current, target, phase="approach", near_grasp=True)
    assert step.clamped
    moved = np.linalg.norm(np.array(step.target.position) - np.array(current.position))
    assert moved <= 0.005 + 1e-9


def test_step_rejects_gross_jump(checker):
    current = Pose((0.15, -0.15, 0.15), (0, 1, 0, 0))
    target = Pose((0.25, -0.15, 0.15), (0, 1, 0, 0))
    with pytest.raises(SafetyError, match="gross"):
        checker.clamp_step(current, target, phase="approach", near_grasp=True)


def test_ik_residual_rejected(checker):
    with pytest.raises(SafetyError, match="residual"):
        checker.check_ik(np.zeros(7), np.zeros(7), pos_error_m=0.01, rot_error_rad=0.0)


def test_joint_jump_rejected(checker):
    q_seed = np.zeros(7)
    q_sol = np.array([0.6, 0, 0, 0, 0, 0, 0])
    with pytest.raises(SafetyError, match="jump"):
        checker.check_ik(q_seed, q_sol, pos_error_m=0.0, rot_error_rad=0.0)


def test_path_below_table_rejected(checker):
    a = Pose((0.15, -0.15, 0.15), (0, 1, 0, 0))
    b = Pose((0.15, -0.15, 0.006), (0, 1, 0, 0))
    with pytest.raises(SafetyError, match="too low"):
        checker.check_path(a, b)
