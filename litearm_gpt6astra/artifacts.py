"""运行产物：result.json / steps.jsonl / ik.json / images / replay.jsonl。"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import cv2
import numpy as np

from .types import Pose


def to_jsonable(value):
    if isinstance(value, Pose):
        return value.as_dict()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


class ArtifactWriter:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.images_dir = self.out_dir / "images"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.steps_path = self.out_dir / "steps.jsonl"
        self.replay_path = self.out_dir / "replay.jsonl"
        self.ik_entries: list[dict] = []
        self.steps_path.write_text("")

    def write_config(self, config, *, astra_info: dict | None = None) -> None:
        payload = to_jsonable(config)
        if astra_info:
            payload["astra_runtime"] = astra_info
        (self.out_dir / "config.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        )

    def save_image(self, name: str, image) -> str:
        path = self.images_dir / name
        cv2.imwrite(str(path), image)
        return str(path.relative_to(self.out_dir))

    def step(self, record: dict) -> None:
        with self.steps_path.open("a") as handle:
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")

    def ik(self, entry: dict) -> None:
        self.ik_entries.append(to_jsonable(entry))

    def replay(self, response: dict) -> None:
        with self.replay_path.open("a") as handle:
            handle.write(json.dumps(to_jsonable(response), ensure_ascii=False) + "\n")

    def finish(self, result) -> None:
        (self.out_dir / "ik.json").write_text(
            json.dumps(
                {"backend": "firmware", "steps": self.ik_entries},
                indent=2,
                ensure_ascii=False,
            )
            + "\n"
        )
        (self.out_dir / "result.json").write_text(
            json.dumps(to_jsonable(result), indent=2, ensure_ascii=False) + "\n"
        )
