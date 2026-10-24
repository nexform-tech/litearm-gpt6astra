"""MuJoCo 相机渲染，输出 BGR uint8。"""

from __future__ import annotations

import time
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class Frame:
    image: np.ndarray
    age_s: float


class Cameras:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self._renderers: dict[tuple[int, int], mujoco.Renderer] = {}

    def _renderer(self, width: int, height: int) -> mujoco.Renderer:
        key = (width, height)
        if key not in self._renderers:
            self._renderers[key] = mujoco.Renderer(self.model, height=height, width=width)
        return self._renderers[key]

    def render(self, camera: str, width: int, height: int) -> Frame:
        started = time.monotonic()
        renderer = self._renderer(width, height)
        renderer.update_scene(self.data, camera=camera)
        rgb = renderer.render()
        bgr = np.ascontiguousarray(rgb[..., ::-1])
        return Frame(image=bgr, age_s=time.monotonic() - started)

    def close(self) -> None:
        for renderer in self._renderers.values():
            renderer.close()
        self._renderers.clear()
