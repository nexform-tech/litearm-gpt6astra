import numpy as np
import pytest

from litearm_gpt6astra.config import load_config
from litearm_gpt6astra.robot.sim import SimBackend


@pytest.fixture
def sim():
    cfg = load_config("configs/sim.yaml")
    with SimBackend(cfg) as s:
        s.reset(seed=0)
        yield s


def test_reset_home_and_block(sim):
    obs = sim.observe(frame_id=0, phase="approach")
    assert np.allclose(obs.arm.q, sim.cfg.home_q, atol=1e-6)
    assert abs(obs.gripper.opening_m - sim.cfg.gripper.open_m) < 2e-3
    bx, by, bz = sim.block_position()
    assert sim.cfg.scene.start_x[0] <= bx <= sim.cfg.scene.start_x[1]
    assert sim.cfg.scene.start_y[0] <= by <= sim.cfg.scene.start_y[1]
    assert abs(bz - sim.cfg.scene.block_size_m / 2) < 1e-6
    assert obs.commanded_tcp.position == pytest.approx(obs.arm.tcp.position, abs=1e-3)


def test_execute_joints(sim):
    q = np.array(sim.cfg.home_q)
    q[0] += 0.05
    result = sim.execute_joints(q)
    assert result.ok
    assert np.max(np.abs(sim.q() - q)) < 0.02


def test_gripper_open_close(sim):
    sim.open_gripper()
    sim.step(500)
    assert sim.gripper_opening_m() > 0.08
    sim.close_gripper()
    sim.step(1000)
    assert sim.gripper_opening_m() < 0.005


def test_tcp_site_offset(sim):
    R = sim.flange_pose().matrix()
    expected = sim.flange_position() + R @ np.array([0, 0, sim.cfg.tcp_offset_m])
    assert np.allclose(sim.tcp_position(), expected, atol=1e-5)


def test_contact_grasp_and_lift(sim):
    from litearm_gpt6astra.robot.ik import Q_MAX, Q_MIN, FirmwareIK
    from litearm_gpt6astra.rotations import matrix_to_quat_wxyz
    from litearm_gpt6astra.types import Pose

    ik = FirmwareIK.from_source()
    Rdown = np.diag([1.0, -1.0, -1.0])
    quat = matrix_to_quat_wxyz(Rdown)
    off = np.array([0, 0, sim.cfg.tcp_offset_m])
    block = sim.block_position().copy()

    def goto(tcp, step):
        nonlocal block
        cur = sim.tcp_position().copy()
        n = max(1, int(np.ceil(np.linalg.norm(tcp - cur) / step)))
        for k in range(1, n + 1):
            tgt = cur + (tcp - cur) * k / n
            p = tgt - Rdown @ off
            q, ok = ik.solve_local(p, Rdown, sim.q(), Q_MIN, Q_MAX, max_joint_jump=1.0)
            assert ok
            result = sim.execute_joints(q, tcp_target=Pose(tuple(tgt), quat))
            assert result.ok, result.error

    goto(block + np.array([0, 0, 0.10]), 0.01)
    goto(block + np.array([0, 0, 0.015]), 0.003)
    sim.close_gripper()
    sim.step(1000)
    before = sim.block_position().copy()
    goto(block + np.array([0, 0, 0.10]), 0.01)
    sim.step(200)
    after = sim.block_position().copy()
    assert after[2] > before[2] + 0.05
