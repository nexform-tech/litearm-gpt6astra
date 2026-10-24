import pytest

from litearm_gpt6astra.config import load_config
from litearm_gpt6astra.robot.sim import SimBackend
from litearm_gpt6astra.robot.task import PickPlaceTask


@pytest.fixture
def task():
    cfg = load_config("configs/sim.yaml")
    with SimBackend(cfg) as sim:
        task = PickPlaceTask(cfg, sim)
        task.reset(seed=0)
        yield task


def test_reset_places_block_in_start_zone(task):
    p = task.block_position()
    assert 0.09 <= p[0] <= 0.15
    assert -0.20 <= p[1] <= -0.15
    assert abs(p[2] - 0.015) < 1e-3


def test_verify_fails_when_block_not_in_target(task):
    r = task.verify(check_image=False)
    assert not r.success
    assert r.reason == "outside_target"


def test_verify_success_after_teleport(task):
    task.sim.set_block_position((0.20, -0.10, 0.015))
    task.sim.step(50)
    r = task.verify(check_image=False)
    assert r.success
    assert r.score == 1.0
