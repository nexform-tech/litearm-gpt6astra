import numpy as np

from litearm_gpt6astra.config import load_config
from litearm_gpt6astra.robot.ik import Q_MAX, Q_MIN


def test_model_names(sim_model):
    bodies = [sim_model.body(i).name for i in range(sim_model.nbody)]
    for name in (
        "base_link",
        "link1",
        "link7",
        "ee_link",
        "gripper_base",
        "finger_left",
        "finger_right",
        "table",
        "block",
    ):
        assert name in bodies
    joints = [sim_model.joint(i).name for i in range(sim_model.njnt)]
    for i in range(1, 8):
        assert f"j{i}" in joints
    assert "finger_right" in joints and "finger_left" in joints
    assert "block_free" in joints
    assert [sim_model.site(i).name for i in range(sim_model.nsite)] == ["flange", "tcp"]
    assert [sim_model.camera(i).name for i in range(sim_model.ncam)] == ["cam_head", "cam_wrist"]
    assert sim_model.nq == 7 + 2 + 7


def test_wrist_camera_moves_with_gripper(sim_model):
    assert (
        sim_model.cam_bodyid[sim_model.camera("cam_wrist").id] == sim_model.body("gripper_base").id
    )


def test_joint_limits_match_urdf(sim_model):
    for i in range(7):
        lo, hi = sim_model.jnt_range[sim_model.joint(f"j{i + 1}").id]
        assert lo <= Q_MIN[i] + 1e-4
        assert hi >= Q_MAX[i] - 1e-4


def test_fk_matches_firmware(sim_model, ik):
    import mujoco

    data = mujoco.MjData(sim_model)
    cfg = load_config("configs/sim.yaml")
    samples = [cfg.home_q, [0.3, -0.4, 0.5, -0.9, 0.2, 0.3, -0.1]]
    for q in samples:
        data.qpos[:7] = q
        mujoco.mj_forward(sim_model, data)
        p_mj = np.asarray(data.site("flange").xpos)
        p_fw, _ = ik.fk(q)
        assert np.linalg.norm(p_mj - p_fw) < 1e-3


def test_tcp_matches_tool_offset(sim_model):
    import mujoco

    data = mujoco.MjData(sim_model)
    cfg = load_config("configs/sim.yaml")
    data.qpos[:7] = cfg.home_q
    mujoco.mj_forward(sim_model, data)
    R = data.site("flange").xmat.reshape(3, 3)
    expected = data.site("flange").xpos + R @ np.array([0, 0, cfg.tcp_offset_m])
    assert np.allclose(data.site("tcp").xpos, expected, atol=1e-5)


def test_cameras_render(sim_model):
    import mujoco

    data = mujoco.MjData(sim_model)
    cfg = load_config("configs/sim.yaml")
    data.qpos[:7] = cfg.home_q
    mujoco.mj_forward(sim_model, data)
    renderer = mujoco.Renderer(sim_model, 120, 160)
    for cam in ("cam_head", "cam_wrist"):
        renderer.update_scene(data, camera=cam)
        image = renderer.render()
        assert image.shape == (120, 160, 3)
        assert image.std() > 5.0
    renderer.close()
