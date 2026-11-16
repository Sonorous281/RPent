# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Offline contracts for adaptive LIBERO RL tools and run options."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robots.libero_rl import tools
from robots.libero_rl.toolkit import LiberoRLToolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.utils import templates

OPEN_GAP = 0.08
SIZE = 32
NEAR, FAR = 0.01, 10.0
GRIPPER_DOWN = np.array([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])


class FakeEnv:
    """Apply OSC deltas exactly; fingers follow an incremental command."""

    def __init__(
        self,
        *,
        block_gap: float = 0.0,
        success_below_z: float | None = None,
        depth_m: float = 1.0,
    ) -> None:
        self.xyz = np.array([0.1, -0.2, 1.0])
        self.rotation = Rotation.from_matrix(GRIPPER_DOWN)
        self.finger_command = -1.0
        self.gap = OPEN_GAP
        self.block_gap = block_gap
        self.success_below_z = success_below_z
        self.depth_m = depth_m
        self.terminated = self.truncated = False
        self.actions: list[np.ndarray] = []
        self._observe()

    def _observe(self) -> None:
        half = self.gap / 2
        self.last_obs = {
            "states": np.concatenate(
                [self.xyz, self.rotation.as_rotvec(), [half, -half]]
            ),
            "main_images": np.zeros((4, 4, 3), dtype=np.uint8),
        }

    def step(self, action):
        assert not (self.terminated or self.truncated)
        action = np.asarray(action, dtype=np.float64)
        self.actions.append(action)
        self.xyz = self.xyz + action[:3] * tools.POSITION_SCALE_M
        self.rotation = (
            Rotation.from_rotvec(action[3:6] * tools.ROTATION_SCALE_RAD) * self.rotation
        )
        self.finger_command = float(
            np.clip(self.finger_command + 0.5 * np.sign(action[6]), -1, 1)
        )
        self.gap = max(OPEN_GAP * (1 - self.finger_command) / 2, self.block_gap)
        self.terminated = bool(
            self.success_below_z is not None and self.xyz[2] < self.success_below_z
        )
        self._observe()
        return self.last_obs, 0.0, self.terminated, False, {}

    def get_task_language(self) -> str:
        return "put the cube in the bowl"

    def render_camera(self, camera_name, height, width, depth=False):
        buffer = np.full((height, width), (1 - NEAR / self.depth_m) / (1 - NEAR / FAR))
        return np.zeros((height, width, 3), dtype=np.uint8), buffer

    def get_camera_meta(self, camera_name, height, width):
        return {
            "intrinsic_K": [[width, 0, width / 2], [0, height, height / 2], [0, 0, 1]],
            "extrinsic_cam2world": np.eye(4).tolist(),
            "depth_near": NEAR,
            "depth_far": FAR,
        }


@pytest.fixture(autouse=True)
def offline_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "IMAGE_SIZE", SIZE)
    monkeypatch.setattr(
        templates, "default_variables", lambda: {"output_dir": "/offline/output"}
    )


def make_toolkit(
    tmp_path: Path,
    mode: str,
    *,
    env: FakeEnv | None = None,
) -> LiberoRLToolkit:
    return LiberoRLToolkit(
        runtime_kwargs={
            "env": env or FakeEnv(),
            "sam3_client": None,
        },
        dashboard_events=NullDashboardEventSink(),
        observation_mode=mode,
        output_dir=tmp_path / "run",
    )


def test_pure_translation_and_pure_rotation_hold_the_other_pose_part(tmp_path) -> None:
    toolkit = make_toolkit(tmp_path, "adaptive")
    start = toolkit.initial_observation().data["state"]["pose"]

    moved = toolkit.execute_tool("move_to", {"xyz": [0.2, -0.1, 0.9]}).data
    turned = toolkit.execute_tool(
        "move_to", {"target_yaw": 1.2, "target_pitch": 0.3}
    ).data

    assert moved["status"] == turned["status"] == "reached"
    assert moved["pose"]["xyz"] == pytest.approx([0.2, -0.1, 0.9], abs=0.006)
    assert moved["pose"]["yaw"] == pytest.approx(start["yaw"], abs=0.03)
    assert moved["pose"]["pitch"] == pytest.approx(start["pitch"], abs=0.03)
    assert turned["pose"]["xyz"] == pytest.approx(moved["pose"]["xyz"], abs=0.006)
    assert turned["pose"]["yaw"] == pytest.approx(1.2, abs=0.03)
    assert turned["pose"]["pitch"] == pytest.approx(0.3, abs=0.03)
    assert turned["step"] == moved["step"] + 1
    assert set(moved) == {
        "status",
        "step",
        "pose",
        "opening",
        "terminated",
        "truncated",
    }


def test_closing_on_an_object_reports_and_keeps_the_measured_opening(tmp_path) -> None:
    env = FakeEnv(block_gap=0.04)
    toolkit = make_toolkit(tmp_path, "adaptive", env=env)

    closed = toolkit.execute_tool("set_gripper", {"opening": 0.0}).data
    actions_before_lift = len(env.actions)
    lifted = toolkit.execute_tool("move_to", {"xyz": [0.1, -0.2, 1.1]}).data

    assert closed["status"] == "partial"
    assert closed["reason"] == "blocked"
    assert closed["opening"] == pytest.approx(0.5, abs=0.01)
    assert lifted["status"] == "reached"
    assert lifted["opening"] == pytest.approx(0.5, abs=0.01)
    assert all(action[6] == 0 for action in env.actions[actions_before_lift:])


def test_environment_success_ends_motion_and_refuses_later_actions(tmp_path) -> None:
    toolkit = make_toolkit(tmp_path, "adaptive", env=FakeEnv(success_below_z=0.95))

    moved = toolkit.execute_tool("move_to", {"xyz": [0.1, -0.2, 0.9]}).data
    refused = toolkit.execute_tool("set_gripper", {"opening": 1.0}).data
    finished = toolkit.execute_tool("finish", {}).data

    assert moved["status"] == "partial"
    assert moved["reason"] == "terminated"
    assert moved["terminated"] is True
    assert refused["error"] == "episode_ended"
    assert refused["step"] == moved["step"]
    assert finished == {
        "_finish": True,
        "success": True,
        "terminated": True,
        "truncated": False,
    }
    assert toolkit.solved()
