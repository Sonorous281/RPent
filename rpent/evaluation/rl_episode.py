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

"""Shared model-client lifecycle and outcome serialization for RL episodes."""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import NoReturn

from openai import AsyncOpenAI
from pydantic_core import to_jsonable_python

from rpent.planner.api_loop import ApiAgentLoop
from rpent.planner.base import PlannerResult
from rpent.session.planner_capture import end_planner_task, start_planner_task
from rpent.tools.toolkit import Toolkit

EXIT_SETUP_FAILED = 75
_TERMINAL_ERRORS = {
    "tool_arguments",
    "tool_name",
    "response_truncated",
    "context_limit",
    "episode_timeout",
}


class _EpisodeTerminated(SystemExit):
    """Exit requested by the episode supervisor."""


def terminate_episode(signum: int, _frame: FrameType | None) -> NoReturn:
    """Unwind the episode so its resources and trajectory are closed."""
    raise _EpisodeTerminated(128 + signum)


def token_counter(tokenizer_path: Path) -> Callable[[str], int]:
    """Count text tokens with the served model's tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    return lambda text: len(tokenizer.encode(text, add_special_tokens=False))


async def solve(
    planner: ApiAgentLoop,
    client: AsyncOpenAI,
    toolkit: Toolkit,
    *,
    system_prompt: str,
    user_message: str,
    max_turns: int,
) -> PlannerResult:
    """Run the planner and close its model client with the episode."""
    async with client:
        return await planner.solve_async(
            system_prompt=system_prompt,
            user_message=user_message,
            toolkit=toolkit,
            max_turns=max_turns,
        )


def episode_outcome(
    result: PlannerResult, toolkit: Toolkit, max_turns: int, *, truncated: bool = False
) -> dict:
    """Describe a finished episode; success always comes from the environment."""
    success = bool(toolkit.solved())
    terminal_error = result.error_kind in _TERMINAL_ERRORS
    if success:
        reason = "success"
    elif result.error:
        reason = result.error_kind if terminal_error else "error"
    elif truncated:
        reason = "environment_limit"
    elif result.stats["turns_used"] >= max_turns:
        reason = "request_limit"
    else:
        reason = "planner_completed"
    return {
        "success": success,
        "truncated": truncated,
        "terminal": success or not result.error or terminal_error,
        "termination_reason": reason,
        "error": result.error,
        "error_kind": result.error_kind,
    }


def write_transcript(output: Path, result: PlannerResult) -> None:
    transcript = {
        "messages": result.messages,
        "stats": result.stats,
        "finish_result": result.finish_result,
        "error": result.error,
        "error_kind": result.error_kind,
    }
    (output / "planner.json").write_text(
        json.dumps(to_jsonable_python(transcript, bytes_mode="base64"), indent=2) + "\n"
    )


@contextmanager
def capture_episode(toolkit: Toolkit, *, instruction: str, config: dict):
    """Capture one task and finalize it after the caller releases its resources.

    The caller appends the planner result before leaving its ExitStack. Keeping
    capture outside that stack records cleanup failures without swallowing them.
    """
    results = []
    state = start_planner_task(toolkit.state, instruction=instruction, config=config)
    try:
        yield results
    finally:
        result = results[0] if results else None
        exception = sys.exc_info()[1]
        terminated = isinstance(exception, _EpisodeTerminated)
        end_planner_task(
            state,
            result,
            finish_result=result.finish_result if result else None,
            environment_success=bool(toolkit.solved()),
            exception=None if terminated else exception,
            cancelled_reason="operator" if terminated else None,
        )
