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

"""Run one LIBERO RL episode for an RL trainer.

An RL trainer such as RLinf starts this module once per episode on the node
that hosts the simulator, with the OpenAI-compatible endpoint that serves its
policy. The end-effector planner solves the task with a fresh LIBERO
environment and the running SAM3 service, with adaptive observations.
The outcome is written to ``outcome.json`` in ``--output-dir``.
RPent keeps per-process state such as the output directory, so every episode
runs in its own process.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import signal
import sys
from collections.abc import Callable
from pathlib import Path

from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider

from robots.libero_rl.robot_spec import get_robot_spec, get_toolkit
from rpent.dashboard.events import NullDashboardEventSink
from rpent.evaluation.rl_episode import (
    EXIT_SETUP_FAILED,
    capture_episode,
    episode_outcome,
    solve,
    token_counter,
    write_transcript,
)
from rpent.planner.api_loop import ApiAgentLoop
from rpent.planner.compaction import build_history_compaction
from rpent.planner.qwen import QwenChatModel, qwen_model_settings
from rpent.robots.robot_spec import RunConfig
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("libero_rl_episode")


def start_environment(
    args: argparse.Namespace, task: dict, release: contextlib.ExitStack
) -> tuple[Toolkit, RunConfig]:
    """Start the task's scene and return its toolkit and config.

    ``release`` stops the started services and closes the toolkit.
    """
    args.suite, args.task, args.seed = task["suite"], task["task"], task["seed"]
    args.planner = "api"
    args.memory_profile, args.memory_dir, args.memory_repo = "none", None, None
    spec = get_robot_spec()
    config = spec.parse_config(args)
    events = NullDashboardEventSink()
    daemons, runtime = spec.init_runtime(args, args.output_dir, events, {"env", "sam3"})
    for daemon in daemons:
        release.callback(daemon.stop)
    toolkit = get_toolkit(
        runtime_kwargs=runtime,
        dashboard_events=events,
        config=config,
    )
    release.callback(toolkit.close)
    return toolkit, config


def render_prompts(config: RunConfig) -> tuple[str, str]:
    """Return the adaptive system prompt and the opening user message."""
    prompts = get_robot_spec().prompts
    variables = config.prompt_vars
    return prompts.render("system", variables=variables), prompts.render(
        "user", variables=variables
    )


def build_planner(
    *,
    base_url: str,
    model_name: str,
    max_tokens: int,
    request_timeout_s: float,
    episode_timeout_s: float,
    observation_mode: str,
    history_compaction: dict | None,
    count_tokens: Callable[[str], int] | None,
    reasoning_effort: str = "none",
    thinking: bool = True,
) -> tuple[ApiAgentLoop, AsyncOpenAI]:
    """Build the native planner on a policy endpoint and its model client.

    Each model request is sent once, without client retries.
    The history_compaction option holds the tiered compaction settings.
    Qwen requests carry thinking-mode sampling, an explicit thinking switch,
    and an effort tier only when requested.
    """
    if observation_mode != "adaptive":
        raise ValueError("libero_rl supports only adaptive observations")
    client = AsyncOpenAI(
        base_url=base_url, api_key="unused", max_retries=0, timeout=request_timeout_s
    )
    # The planner calls this capability factory once per run, so it must be a
    # callable whenever compaction is enabled and absent otherwise.
    compaction = (
        (lambda: build_history_compaction(history_compaction, tokenizer=count_tokens))
        if history_compaction is not None
        else None
    )
    planner = ApiAgentLoop(
        model=QwenChatModel(
            model_name,
            provider=OpenAIProvider(openai_client=client),
            # SGLang constrains the whole tool set once any tool is strict.
            profile={"openai_supports_strict_tool_definition": False},
        ),
        dashboard_events=NullDashboardEventSink(),
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        thinking=thinking,
        no_images=False,
        timeout_s=episode_timeout_s,
        compaction=compaction,
        model_settings=qwen_model_settings(
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            thinking=thinking,
            parallel_tool_calls=False,
        ),
        sequential_tools=True,
    )
    return planner, client


def run(args: argparse.Namespace) -> int:
    """Run the requested episode, write ``outcome.json`` and return the exit status."""
    request = json.loads(args.request.read_text())
    task, options = request["task"], dict(request["planner"])
    max_turns = options.pop("max_turns")
    history_compaction = options.pop("history_compaction", None)
    if options.pop("no_images", False):
        raise ValueError("libero_rl requires images from view_env")
    args.observation_mode = options.pop("observation_mode", args.observation_mode)
    args.prompt = options.pop("prompt", args.prompt)
    init_output_dir(args.output_dir)
    with contextlib.ExitStack() as capture, contextlib.ExitStack() as release:
        try:
            toolkit, config = start_environment(args, task, release)
        except Exception:
            logger.exception("the episode environment did not start")
            return EXIT_SETUP_FAILED
        system_prompt, user_message = render_prompts(config)
        captured_results = capture.enter_context(
            capture_episode(toolkit, instruction=user_message, config=request)
        )
        count_tokens = (
            token_counter(args.tokenizer_path) if args.tokenizer_path else None
        )
        planner, client = build_planner(
            **options,
            observation_mode=args.observation_mode,
            history_compaction=history_compaction,
            count_tokens=count_tokens,
        )
        result = asyncio.run(
            solve(
                planner,
                client,
                toolkit,
                system_prompt=system_prompt,
                user_message=user_message,
                max_turns=max_turns,
            )
        )
        captured_results.append(result)
        write_transcript(args.output_dir, result)
        outcome = {
            **episode_outcome(result, toolkit, max_turns),
            "observation_mode": args.observation_mode,
            "prompt": args.prompt,
            "init_state_id": task.get("init_state_id"),
            "memory_entries": [],
        }
    # Written once the simulator is released, so an outcome means a clean end.
    (args.output_dir / "outcome.json").write_text(json.dumps(outcome) + "\n")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    get_robot_spec().add_cli_args(parser, use_dashboard=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--request",
        type=Path,
        required=True,
        help="JSON with the episode's task and planner options.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Tokenizer of the served planner model, used to count tokens for "
        "history compaction.",
    )
    args = parser.parse_args()
    # A stopped episode still releases its simulator and model client.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(128 + signal.SIGTERM))
    sys.exit(run(args))


if __name__ == "__main__":
    main()
