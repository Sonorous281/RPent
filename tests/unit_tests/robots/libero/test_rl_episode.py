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

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

from robots.libero import rl_episode
from rpent.dashboard.events import NullDashboardEventSink
from rpent.memory import MemoryManager
from rpent.session import EnvState
from rpent.tools import Toolkit, ToolResult, tool
from tests.e2e_tests.offline_planner_server import (
    OfflinePlannerServer,
    ScriptedToolCall,
)


class RobotToolkit(Toolkit):
    """A deterministic robot boundary, using the real tool dispatcher and state."""

    def __init__(self, path):
        super().__init__(
            dashboard_events=NullDashboardEventSink(),
            state=EnvState(path),
            memory=MemoryManager(path / "memory"),
        )
        self.get_env_state()

    def solved(self) -> bool:
        return False

    def get_env_state(self, **kwargs):
        with self.state.record_step(state={}, terminated=False):
            pass
        return ToolResult(data={})


@pytest.fixture
def toolkit(tmp_path, monkeypatch):
    monkeypatch.setattr("rpent.utils.templates.get_output_dir", lambda: tmp_path)
    toolkit = RobotToolkit(tmp_path)
    yield toolkit
    toolkit.close()


def run_planner(toolkit, base_url, max_turns=3, **options):
    """Run the episode planner against ``base_url`` and return its outcome."""
    settings = {
        "max_tokens": 64,
        "request_timeout_s": 600,
        "episode_timeout_s": 1200,
        "no_images": False,
        "history_compaction": None,
        **options,
    }
    planner, client = rl_episode.build_planner(
        base_url=base_url, model_name="policy", count_tokens=None, **settings
    )
    result = asyncio.run(
        rl_episode.solve(
            planner,
            client,
            toolkit,
            system_prompt="Move the robot.",
            user_message="Move the robot.",
            max_turns=max_turns,
        )
    )
    return result, rl_episode.episode_outcome(result, toolkit, max_turns)


def test_episode_deadline_drains_tools_and_returns_native_failure(toolkit, tmp_path):
    stopped = threading.Event()

    @tool
    def slow_move() -> ToolResult:
        """Move until the episode budget interrupts the robot."""
        try:
            while True:
                toolkit.raise_if_cancelled()
                time.sleep(0.005)
        finally:
            stopped.set()

    toolkit.add_tool(slow_move)
    with OfflinePlannerServer((ScriptedToolCall("slow_move", {}),)) as endpoint:
        result, status = run_planner(toolkit, endpoint.base_url, episode_timeout_s=1)
        assert endpoint.request_count == 1 and stopped.is_set()
    assert status["terminal"] and not status["success"]
    assert status["termination_reason"] == status["error_kind"] == "episode_timeout"
    rl_episode.write_transcript(tmp_path, result)
    recorded = json.loads((tmp_path / "planner.json").read_text())
    assert recorded["stats"]["turns_used"] == 1
    assert recorded["error_kind"] == "episode_timeout"
