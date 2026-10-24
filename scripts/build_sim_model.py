#!/usr/bin/env python3
"""Build the MuJoCo scene from the arm URDF and gripper xacro."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from litearm_gpt6astra.model.urdf_to_mjcf import build_model  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm-urdf", type=Path, required=True)
    parser.add_argument("--gripper-xacro", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--tcp-offset-m", type=float, default=0.110)
    args = parser.parse_args()
    out = build_model(
        output=args.output,
        arm_urdf=args.arm_urdf,
        gripper_xacro=args.gripper_xacro,
        tcp_offset_m=args.tcp_offset_m,
    )
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
