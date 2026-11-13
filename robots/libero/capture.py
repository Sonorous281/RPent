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

"""Executed LIBERO actions in the shared trajectory capture format."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from robots.libero.env_client import LiberoEnvClient


def capture_libero_call(
    env: LiberoEnvClient,
    method: Literal["step", "chunk_step"],
    actions: Any,
    *,
    return_all_frames: bool | None = None,
) -> dict[str, Any]:
    """Execute a LIBERO call and collect its actions and observations."""
    before = env.last_obs
    if method == "step":
        result = env.step(actions)
        actions = [np.asarray(actions)]
        steps = [result]
    else:
        if return_all_frames is None:
            return_all_frames = env.return_all_frames
        obs, rewards, term, trunc, infos = env.chunk_step(
            actions, return_all_frames=True
        )
        result = (obs if return_all_frames else obs[-1], rewards, term, trunc, infos)
        actions = np.asarray(actions)
        steps = zip(obs, rewards, term, trunc, infos)

    def observation_payload(obs):
        return {
            "format": "libero.observation",
            "data": obs,
            "completeness": "complete",
            "inline_array_paths": ("/states",),
        }

    return {
        "result": result,
        "before_observation": observation_payload(before),
        "before_observation_cached": True,
        "steps": [
            {
                "action": {
                    "format": "libero.action",
                    "data": action,
                    "completeness": "complete",
                    "inline_array_paths": ("",),
                },
                "after_observation": observation_payload(obs),
                "result": {
                    "format": "libero.step_result",
                    "data": {
                        "reward": reward,
                        "terminated": term,
                        "truncated": trunc,
                        "info": info,
                    },
                    "completeness": "complete",
                    "inline_array_paths": (
                        "/info/episode/success_once",
                        "/info/episode/return",
                        "/info/episode/episode_len",
                        "/info/episode/reward",
                    ),
                },
            }
            for action, (obs, reward, term, trunc, info) in zip(actions, steps)
        ],
    }
