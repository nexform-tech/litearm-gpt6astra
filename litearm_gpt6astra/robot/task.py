"""pick_place 的复位与验证。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import AppConfig
from ..types import VerificationResult
from .safety import SafetyChecker
from .sim import SimBackend

RED_HSV = ((0, 120, 80), (10, 255, 255))
RED_HSV_WRAP = ((170, 120, 80), (179, 255, 255))
GREEN_HSV = ((45, 80, 60), (85, 255, 255))


@dataclass
class Blob:
    centroid_px: tuple[float, float]
    area_px: float
    bbox: tuple[int, int, int, int]


def find_blob(image_bgr, lower, upper, *, wrap=None, min_area: float = 30.0) -> Blob | None:
    if image_bgr is None or image_bgr.size == 0 or image_bgr.ndim != 3:
        return None
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    if wrap is not None:
        mask |= cv2.inRange(hsv, np.array(wrap[0], np.uint8), np.array(wrap[1], np.uint8))
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < min_area:
        return None
    M = cv2.moments(contour)
    if abs(M["m00"]) < 1e-9:
        return None
    cx = float(M["m10"] / M["m00"])
    cy = float(M["m01"] / M["m00"])
    x, y, w, h = cv2.boundingRect(contour)
    return Blob((cx, cy), area, (int(x), int(y), int(w), int(h)))


class PickPlaceTask:
    def __init__(self, cfg: AppConfig, sim: SimBackend, safety: SafetyChecker | None = None):
        self.cfg = cfg
        self.sim = sim
        self.safety = safety or SafetyChecker(cfg.safety)
        self._last_block: np.ndarray | None = None

    def reset(self, seed: int = 0) -> None:
        self.sim.reset(seed)
        self.safety.check_position(self.sim.tcp_position())
        self.sim.open_gripper()
        self.sim.step(200)
        self._last_block = None

    def block_position(self) -> np.ndarray:
        return self.sim.block_position()

    def target_center(self) -> np.ndarray:
        return np.array([self.cfg.scene.target_center[0], self.cfg.scene.target_center[1], 0.0])

    def observe_images(self) -> dict[str, np.ndarray]:
        images = {}
        for name, cam in (
            ("head", self.cfg.cameras.head),
            ("wrist", self.cfg.cameras.wrist),
        ):
            images[name] = self.sim.cameras.render(cam.name, cam.width, cam.height).image
        return images

    # ------------------------------------------------------------- projection
    def project(self, camera: str, point, width: int, height: int) -> tuple[float, float] | None:
        cam_id = self.sim.model.camera(camera).id
        p = np.asarray(point, dtype=float)
        cam_pos = np.asarray(self.sim.data.cam_xpos[cam_id], dtype=float)
        cam_mat = np.asarray(self.sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
        local = cam_mat.T @ (p - cam_pos)
        if local[2] >= -1e-6:
            return None
        fovy = float(self.sim.model.cam_fovy[cam_id]) * math.pi / 180.0
        fy = (height / 2.0) / math.tan(fovy / 2.0)
        fx = fy
        u = width / 2.0 + fx * local[0] / (-local[2])
        v = height / 2.0 - fy * local[1] / (-local[2])
        return float(u), float(v)

    def pixel_to_table(
        self, camera: str, u: float, v: float, z_plane: float, width: int, height: int
    ):
        """像素射线与水平面 z=z_plane 的交点，返回基座系 (x, y)。"""
        cam_id = self.sim.model.camera(camera).id
        cam_pos = np.asarray(self.sim.data.cam_xpos[cam_id], dtype=float)
        R = np.asarray(self.sim.data.cam_xmat[cam_id], dtype=float).reshape(3, 3)
        fovy = float(self.sim.model.cam_fovy[cam_id]) * math.pi / 180.0
        fy = (height / 2.0) / math.tan(fovy / 2.0)
        fx = fy
        d_cam = np.array([(u - width / 2.0) / fx, -(v - height / 2.0) / fy, -1.0])
        d_world = R @ d_cam
        if abs(float(d_world[2])) < 1e-9:
            return None
        t = (z_plane - cam_pos[2]) / float(d_world[2])
        if t <= 0:
            return None
        point = cam_pos + t * d_world
        return [float(point[0]), float(point[1])]

    def visual_candidates(self, images: dict[str, np.ndarray]) -> dict:
        """颜色候选 + 桌面平面近似 XY。不是深度，也不是精确抓取位姿。"""
        out = {}
        for name, cam in (("head", self.cfg.cameras.head), ("wrist", self.cfg.cameras.wrist)):
            image = images.get(name)
            if image is None:
                continue
            red = find_blob(image, *RED_HSV, wrap=RED_HSV_WRAP)
            green = find_blob(image, *GREEN_HSV, min_area=20.0)
            entry = {
                "use_for_alignment": name == "head",
                "primary": name == "head",
                "tcp_px": self.project(cam.name, self.sim.tcp_position(), cam.width, cam.height),
            }
            if red is not None:
                entry["red_cube_px"] = list(red.centroid_px)
                entry["red_cube_bbox"] = list(red.bbox)
                entry["red_cube_table_xy"] = self.pixel_to_table(
                    cam.name,
                    red.centroid_px[0],
                    red.centroid_px[1],
                    self.cfg.scene.block_size_m / 2.0,
                    cam.width,
                    cam.height,
                )
            if green is not None:
                entry["green_zone_px"] = list(green.centroid_px)
                entry["green_zone_table_xy"] = self.pixel_to_table(
                    cam.name,
                    green.centroid_px[0],
                    green.centroid_px[1],
                    0.0,
                    cam.width,
                    cam.height,
                )
            out[name] = entry
        return out

    # ---------------------------------------------------------------- verify
    def verify(self, *, check_image: bool = True) -> VerificationResult:
        block = self.block_position()
        target = self.target_center()
        distance = float(np.linalg.norm(block[:2] - target[:2]))
        radius = float(self.cfg.scene.target_radius_m)
        on_table = abs(float(block[2]) - self.cfg.scene.block_size_m / 2.0) <= 0.01
        if self._last_block is None:
            self.sim.step(int(0.1 / self.cfg.scene.timestep))
            self._last_block = self.sim.block_position().copy()
            block = self.block_position()
        stable = float(np.linalg.norm(block[:2] - self._last_block[:2])) < 0.001
        self._last_block = block.copy()

        opening = self.sim.gripper_opening_m()
        released = opening >= 0.02
        metrics = {
            "block_position": [float(v) for v in block],
            "target_center": [float(v) for v in target],
            "distance_m": distance,
            "z_m": float(block[2]),
            "stable": stable,
            "gripper_opening_m": opening,
        }

        image_ok = not check_image
        if check_image:
            images = self.observe_images()
            per_view = {}
            for name, cam in (("wrist", self.cfg.cameras.wrist), ("head", self.cfg.cameras.head)):
                image = images[name]
                red = find_blob(image, *RED_HSV, wrap=RED_HSV_WRAP)
                green = find_blob(image, *GREEN_HSV, min_area=20.0)
                projected = self.project(cam.name, block, cam.width, cam.height)
                px_err = None
                if red is not None and projected is not None:
                    px_err = math.hypot(
                        red.centroid_px[0] - projected[0],
                        red.centroid_px[1] - projected[1],
                    )
                ok = red is not None and green is not None and (px_err is None or px_err < 80.0)
                per_view[name] = {
                    "red": None if red is None else {"centroid_px": red.centroid_px},
                    "green": None if green is None else {"centroid_px": green.centroid_px},
                    "block_projected_px": projected,
                    "red_px_error": px_err,
                    "ok": ok,
                }
                if ok:
                    image_ok = True
            metrics["views"] = per_view
            if not image_ok:
                metrics["image_reason"] = "no view has red block + green target"

        if not on_table:
            reason = "block_not_on_table"
        elif not released:
            reason = "gripper_closed"
        elif distance > radius:
            reason = "outside_target"
        elif not stable:
            reason = "unstable"
        elif not image_ok:
            reason = "block_not_found"
        else:
            reason = None

        success = reason is None
        score = 1.0 if success else max(0.0, 1.0 - distance / (4.0 * radius))
        return VerificationResult(
            success=success,
            score=score,
            reason=reason,
            metrics=metrics,
        )
