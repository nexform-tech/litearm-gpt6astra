import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest


@pytest.fixture(scope="session")
def ik():
    from litearm_gpt6astra.robot.ik import FirmwareIK

    return FirmwareIK.from_source()


@pytest.fixture(scope="session")
def sim_model():
    import mujoco

    path = ROOT / "assets" / "sim_scene.xml"
    return mujoco.MjModel.from_xml_path(str(path))
