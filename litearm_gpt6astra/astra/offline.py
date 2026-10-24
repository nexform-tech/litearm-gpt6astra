"""离线策略：脚本策略、回放、以及 live 客户端的薄包装。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..types import Observation
from .client import AstraClient, ChatCompletionsClient, Halt
from .prompt import INSTRUCTIONS


def _proposal(
    status,
    target,
    *,
    gripper,
    near_grasp=False,
    between=False,
    tips=False,
    in_zone=False,
    reason="scripted",
):
    return {
        "status": status,
        "tcp_target": {
            "position": [float(v) for v in target.position],
            "quaternion_wxyz": [float(v) for v in target.quaternion_wxyz],
        },
        "gripper": int(gripper),
        "confidence": 0.95,
        "target_visible": True,
        "hand_visible": True,
        "path_clear": True,
        "near_grasp": bool(near_grasp),
        "object_between_fingers": bool(between),
        "tips_at_grasp_height": bool(tips),
        "object_in_target_zone": bool(in_zone),
        "uncertain": False,
        "reason": reason,
    }


def _step_toward(current: np.ndarray, goal: np.ndarray, max_step: float) -> np.ndarray:
    delta = goal - current
    dist = float(np.linalg.norm(delta))
    if dist <= max_step or dist < 1e-12:
        return goal
    return current + delta / dist * max_step


class ScriptedPolicy:
    """不联网的确定性抓取放置，用来验证机械/闭环/产物。"""

    def __init__(self):
        self.stage = "pre"
        self.lift_target = None

    def propose(self, obs: Observation, sim, phase: str, observation_json=None, views=None):
        block = np.asarray(sim.block_position(), dtype=float)
        target = np.array([sim.cfg.scene.target_center[0], sim.cfg.scene.target_center[1], 0.0])
        current = np.asarray(obs.commanded_tcp.position, dtype=float)
        quat = obs.commanded_tcp.quaternion_wxyz
        coarse = float(sim.cfg.safety.max_step_m)
        fine = float(sim.cfg.safety.max_step_near_grasp_m)

        if phase in ("done", "held"):
            return _proposal("hold", obs.commanded_tcp, gripper=obs.gripper.closure >= 0.5)

        if phase == "approach":
            pre = block + np.array([0.0, 0.0, 0.10])
            grasp = block + np.array([0.0, 0.0, self._grasp_dz()])
            if self.stage == "pre":
                if np.linalg.norm(current - pre) < 0.006:
                    self.stage = "descend"
                else:
                    goal = _step_toward(current, pre, coarse)
                    return self._move(goal, quat, gripper=0, near_grasp=False)
            if self.stage == "descend":
                if np.linalg.norm(current - grasp) < 0.004:
                    self.stage = "grasp"
                    return self._grasp(obs, between=True, tips=True)
                near = np.linalg.norm(current - grasp) < 0.01
                goal = _step_toward(current, grasp, fine if near else coarse)
                return self._move(goal, quat, gripper=0, near_grasp=near)
            return self._grasp(obs, between=True, tips=True)

        if phase == "pregrasp":
            return self._grasp(obs, between=True, tips=True)

        if phase == "transfer":
            above = np.array([target[0], target[1], 0.15])
            place = np.array([target[0], target[1], 0.05])
            if self.stage in ("grasp", "pre", "descend"):
                self.stage = "lift"
                self.lift_target = current + np.array([0.0, 0.0, 0.10])
            if self.stage == "lift":
                if np.linalg.norm(current - self.lift_target) < 0.006:
                    self.stage = "transfer"
                else:
                    goal = _step_toward(current, self.lift_target, coarse)
                    return self._move(goal, quat, gripper=1, near_grasp=False)
            if self.stage == "transfer":
                if np.linalg.norm(current - above) < 0.006:
                    self.stage = "place"
                else:
                    goal = _step_toward(current, above, coarse)
                    return self._move(goal, quat, gripper=1, near_grasp=False)
            if self.stage == "place":
                if np.linalg.norm(current - place) < 0.005:
                    self.stage = "release"
                    return self._release(obs)
                near = np.linalg.norm(current - place) < 0.01
                goal = _step_toward(current, place, fine if near else coarse)
                return self._move(goal, quat, gripper=1, near_grasp=near)
            return self._release(obs)

        if phase == "released":
            return _proposal("done", obs.commanded_tcp, gripper=0, reason="place verified")

        return _proposal("hold", obs.commanded_tcp, gripper=obs.gripper.closure >= 0.5)

    def _grasp_dz(self) -> float:
        return 0.015

    def _grasp(self, obs, *, between, tips):
        return _proposal(
            "grasp",
            obs.commanded_tcp,
            gripper=1,
            near_grasp=True,
            between=between,
            tips=tips,
            reason="cube between fingers",
        )

    def _release(self, obs):
        return _proposal(
            "release",
            obs.commanded_tcp,
            gripper=0,
            near_grasp=True,
            in_zone=True,
            reason="cube over target zone",
        )

    def _move(self, position, quat, *, gripper, near_grasp):
        from ..types import Pose

        return _proposal(
            "move",
            Pose(tuple(float(v) for v in position), quat),
            gripper=gripper,
            near_grasp=near_grasp,
        )


class ReplayAstra:
    """回放预录的 Astra 响应，一行一个 JSON。"""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.responses = [
            json.loads(line) for line in self.path.read_text().splitlines() if line.strip()
        ]
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    def propose(self, obs, sim, phase, observation_json=None, views=None):
        if self.index >= len(self.responses):
            raise Halt("replay exhausted")
        response = self.responses[self.index]
        self.index += 1
        return response


class LivePolicy:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        key_env: str,
        timeout_s: float = 30.0,
        proxy: str = "",
        api_style: str = "chat",
    ):
        client_cls = ChatCompletionsClient if api_style == "chat" else AstraClient
        self.client = client_cls(
            model,
            base_url,
            key_env,
            timeout_s=timeout_s,
            proxy=proxy,
            instructions=INSTRUCTIONS,
        )

    @property
    def key_present(self) -> bool:
        return self.client.key_present

    def propose(self, obs, sim, phase, observation_json=None, views=None):
        if observation_json is None or not views:
            raise Halt("live policy requires observation JSON and camera views")
        return self.client.decide(observation_json, views)
