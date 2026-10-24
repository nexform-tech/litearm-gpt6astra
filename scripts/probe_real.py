#!/usr/bin/env python3
"""Read-only probe for real LiteArm + LiteGrip.

Connects, reads firmware/state/TCP, reads one gripper status frame, and
optionally grabs one camera frame. It never enables, resets, homes, or moves the
arm, and it never enables or disables the gripper.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def probe_arm(port: str | None) -> dict:
    import litearm as pa

    arm = pa.Arm(port=port).connect()
    try:
        state_msg = arm.get_state(timeout=1.0)
        state = state_msg.value
        tcp_msg = arm.get_tcp(timeout=1.0)
        tcp = tcp_msg.value
        return {
            "firmware": arm.firmware,
            "n": arm.n,
            "mode": getattr(state, "mode_name", None) if state else None,
            "flags": list(getattr(state, "flag_names", [])) if state else None,
            "enabled": getattr(state, "enabled", None) if state else None,
            "faulted": getattr(state, "faulted", None) if state else None,
            "q": list(getattr(state, "q", [])) if state else None,
            "tcp": list(tcp) if tcp else None,
            "state_hz": float(getattr(state_msg, "hz", 0.0)),
            "tcp_hz": float(getattr(tcp_msg, "hz", 0.0)),
        }
    finally:
        arm.close()


def probe_gripper(channel: str) -> dict:
    from litegrip import LiteGrip

    # Read-only: do not disable the motor on disconnect.
    with LiteGrip(channel, disable_on_disconnect=False) as gripper:
        loaded = gripper.load_calibration()
        gripper.refresh_status()
        state = gripper.get_state(wait=True)
        return {
            "calibration_loaded": bool(loaded),
            "position_mm": float(state.position_mm),
            "force_n": float(state.force_n),
            "error_code": int(state.error_code),
            "fresh": bool(state.fresh),
            "age_s": state.data_age_s,
            "stale": bool(state.is_stale()),
        }


def probe_camera(device: str, out_dir: Path) -> dict:
    import cv2

    source = int(device) if device.isdigit() else device
    capture = cv2.VideoCapture(source)
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open camera {device}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"cannot read camera {device}")
        path = out_dir / "camera.jpg"
        cv2.imwrite(str(path), frame)
        return {
            "device": str(device),
            "shape": list(frame.shape),
            "image": str(path),
        }
    finally:
        capture.release()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default=None, help="arm serial port; auto-discover if omitted")
    parser.add_argument("--can", default="can0", help="gripper CAN interface")
    parser.add_argument("--camera", default=None, help="camera device, e.g. /dev/video0 or 0")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--skip-arm", action="store_true")
    parser.add_argument("--skip-gripper", action="store_true")
    parser.add_argument("--skip-camera", action="store_true")
    args = parser.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out or f"runs/real_probe_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    report: dict = {
        "timestamp": stamp,
        "arm": None,
        "gripper": None,
        "camera": None,
        "errors": [],
    }

    if not args.skip_arm:
        try:
            report["arm"] = probe_arm(args.port)
        except Exception as exc:  # noqa: BLE001 - probe reports every failure
            report["errors"].append(f"arm: {type(exc).__name__}: {exc}")

    if not args.skip_gripper:
        try:
            report["gripper"] = probe_gripper(args.can)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"gripper: {type(exc).__name__}: {exc}")

    if args.camera and not args.skip_camera:
        try:
            report["camera"] = probe_camera(args.camera, out_dir)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"camera: {type(exc).__name__}: {exc}")

    path = out_dir / "probe.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"report: {path}")
    return 0 if not report["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
