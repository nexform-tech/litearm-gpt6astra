import pytest

from litearm_gpt6astra.config import load_config


def test_load_repo_sim_config():
    cfg = load_config("configs/sim.yaml")
    assert cfg.tcp_offset_m == 0.110
    assert cfg.home_q == (-0.813, -0.523, -0.970, -2.136, 2.532, 1.340, -0.168)
    assert cfg.safety.workspace_z == (0.005, 0.65)
    assert cfg.astra.model == "gpt-6.1-sol"
    assert cfg.astra.api_style == "chat"
    assert cfg.astra.base_url == "https://mmu.vod-qcloud.com/v1"


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("nope: 1\n")
    with pytest.raises(ValueError, match="unknown"):
        load_config(p)


def test_missing_key_rejected(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("seed: 0\n")
    with pytest.raises(ValueError, match="missing"):
        load_config(p)


def test_bool_is_not_number(tmp_path):
    p = tmp_path / "bad.yaml"
    with open("configs/sim.yaml") as handle:
        text = handle.read().replace("timestep: 0.002", "timestep: true")
    p.write_text(text)
    with pytest.raises(ValueError, match="finite number"):
        load_config(p)
