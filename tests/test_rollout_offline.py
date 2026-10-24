import json

from litearm_gpt6astra.astra.offline import ReplayAstra, ScriptedPolicy
from litearm_gpt6astra.config import load_config
from litearm_gpt6astra.rollout import Rollout


def test_scripted_pick_place_succeeds(tmp_path):
    cfg = load_config("configs/sim.yaml")
    rollout = Rollout(cfg, policy=ScriptedPolicy(), out_dir=tmp_path, seed=0)
    result = rollout.run()
    assert result.success
    assert result.phase == "done"
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "steps.jsonl").is_file()
    assert (tmp_path / "ik.json").is_file()
    steps = [json.loads(line) for line in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert len(steps) >= 12
    for step in steps:
        assert "proposal" in step
        assert "executed" in step


def test_replay_reproduces_success(tmp_path):
    cfg = load_config("configs/sim.yaml")
    first = tmp_path / "first"
    Rollout(cfg, policy=ScriptedPolicy(), out_dir=first, seed=0).run()
    steps = [json.loads(line) for line in (first / "steps.jsonl").read_text().splitlines()]
    replay_file = tmp_path / "replay.jsonl"
    with replay_file.open("w") as handle:
        for step in steps:
            if step.get("proposal") is not None:
                handle.write(json.dumps(step["proposal"]) + "\n")
    second = tmp_path / "second"
    result = Rollout(
        cfg,
        policy=ReplayAstra(replay_file),
        out_dir=second,
        seed=0,
        mode="replay",
    ).run()
    assert result.success


def test_low_confidence_latches_held(tmp_path):
    cfg = load_config("configs/sim.yaml")
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        json.dumps(
            {
                "status": "move",
                "tcp_target": {
                    "position": [0.15, -0.15, 0.145],
                    "quaternion_wxyz": [0, 1, 0, 0],
                },
                "gripper": 0,
                "confidence": 0.5,
                "target_visible": True,
                "hand_visible": True,
                "path_clear": True,
                "near_grasp": False,
                "object_between_fingers": False,
                "tips_at_grasp_height": False,
                "object_in_target_zone": False,
                "uncertain": False,
                "reason": "test",
            }
        )
        + "\n"
    )
    result = Rollout(
        cfg, policy=ReplayAstra(bad), out_dir=tmp_path / "run", seed=0, mode="replay"
    ).run()
    assert not result.success
    assert result.phase == "held"


def test_stop_latches_held(tmp_path):
    cfg = load_config("configs/sim.yaml")
    stop = tmp_path / "stop.jsonl"
    stop.write_text(
        json.dumps(
            {
                "status": "stop",
                "tcp_target": {
                    "position": [0.15, -0.15, 0.15],
                    "quaternion_wxyz": [0, 1, 0, 0],
                },
                "gripper": 0,
                "confidence": 0.99,
                "target_visible": False,
                "hand_visible": True,
                "path_clear": True,
                "near_grasp": False,
                "object_between_fingers": False,
                "tips_at_grasp_height": False,
                "object_in_target_zone": False,
                "uncertain": True,
                "reason": "target lost",
            }
        )
        + "\n"
    )
    result = Rollout(
        cfg, policy=ReplayAstra(stop), out_dir=tmp_path / "run", seed=0, mode="replay"
    ).run()
    assert not result.success
    assert result.phase == "held"
    assert any("target lost" in error or "STOP" in error for error in result.errors)
