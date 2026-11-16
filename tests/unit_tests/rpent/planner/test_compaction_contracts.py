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

"""Image accounting, retention, and prompt preservation for planner compaction."""

from __future__ import annotations

import io

import pytest
from pydantic_ai import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from rpent.planner.compaction import (
    ImageAwareSlidingWindowCompaction,
    count_image_tokens_in,
    estimate_request_tokens,
    preserve_prompts,
    retain_images,
    retained_image_keys,
)


def png(width: int, height: int) -> bytes:
    """Return a solid PNG of the given size."""
    PIL = pytest.importorskip("PIL.Image")
    buffer = io.BytesIO()
    PIL.new("RGB", (width, height)).save(buffer, format="PNG")
    return buffer.getvalue()


def image(width: int = 1024, height: int = 1024) -> BinaryContent:
    return BinaryContent(data=png(width, height), media_type="image/png")


def observation(call_id: str, images: int = 2) -> list:
    """A tool call with its image-bearing result, as a rollout planner sees it."""
    return [
        ModelResponse(
            parts=[ToolCallPart("view_env", {}, tool_call_id=call_id)],
        ),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    "view_env",
                    [image() for _ in range(images)],
                    tool_call_id=call_id,
                )
            ]
        ),
    ]


def test_image_cost_is_added_to_the_text_estimate():
    raw = bytes(image(1024, 1024).data)
    history = [ModelRequest(parts=[UserPromptPart(content=["go", image()])])]
    text_only = estimate_request_tokens(history) - count_image_tokens_in(history)
    assert count_image_tokens_in(history) == 1024
    # The text estimate must not be inflated by the image payload's characters.
    assert text_only < 100
    assert len(raw) > 1000


def test_retention_keeps_every_image_inside_the_recent_pairs():
    # Two recent pairs carry images, so both survive and the older one does not.
    history = [
        ModelRequest(parts=[UserPromptPart("task")]),
        *observation("old"),
        *observation("new1"),
        *observation("new2"),
    ]
    keep = retained_image_keys(history, keep_pairs=2)
    assert {key for _, key in keep} == {"new1", "new2"}

    kept = retain_images(history, keep)
    assert count_image_tokens_in(kept) == 1024 * 4  # two pairs x two images
    assert "[image omitted during compaction]" in str(kept[2].parts[0].content)


def test_retention_falls_back_to_the_latest_image_when_recent_pairs_have_none():
    # The recent pair is a text-only tool result, so only one older image
    # survives instead of the whole stale sequence.
    history = [
        ModelRequest(parts=[UserPromptPart("task")]),
        *observation("img1"),
        *observation("img2"),
        ModelResponse(parts=[ToolCallPart("move_to", {}, tool_call_id="text")]),
        ModelRequest(
            parts=[
                ToolReturnPart("move_to", {"status": "reached"}, tool_call_id="text")
            ]
        ),
    ]
    keep = retained_image_keys(history, keep_pairs=1)
    assert len(keep) == 1

    kept = retain_images(history, keep)
    assert count_image_tokens_in(kept) == 1024 * 2  # the latest pair's two images


def test_preserve_prompts_restores_each_exactly_once():
    system = SystemPromptPart("you control one arm")
    user = UserPromptPart("the task follows")
    original = [
        ModelRequest(parts=[system, user]),
        ModelResponse(parts=[TextPart("ok")]),
    ]

    # A summarizing pass dropped both prompts.
    compacted = [ModelRequest(parts=[SystemPromptPart("<summary>did things")])]
    restored = preserve_prompts(original, compacted)

    systems = [
        part
        for message in restored
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, SystemPromptPart) and part.content == "you control one arm"
    ]
    users = [
        part
        for message in restored
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and part.content == "the task follows"
    ]
    assert len(systems) == 1
    assert len(users) == 1


@pytest.mark.anyio
async def test_message_window_triggers_on_images_without_dropping_the_task():
    from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import RunContext
    from pydantic_ai.usage import RunUsage
    from pydantic_ai_harness.compaction._shared import estimate_context_tokens

    history = [
        ModelRequest(parts=[UserPromptPart("pick up the cup")]),
        ModelResponse(parts=[TextPart("looking")]),
        ModelRequest(parts=[UserPromptPart(["old view", image()])]),
        ModelResponse(parts=[TextPart("moving")]),
        ModelRequest(parts=[UserPromptPart(["current view", image()])]),
    ]
    # Plain message counting misses the image tokens and would not compact.
    assert estimate_context_tokens(history) < 1500
    assert estimate_request_tokens(history) > 1500
    model = TestModel()
    request = ModelRequestContext(
        model=model,
        messages=history,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    strategy = ImageAwareSlidingWindowCompaction(
        max_tokens=1500, keep_messages=2, receipts=False
    )
    result = await strategy.before_model_request(
        RunContext(deps=None, model=model, usage=RunUsage()), request
    )
    assert result.messages[0] == history[0]
    assert result.messages[-1] == history[-1]
    assert count_image_tokens_in(result.messages) == 1024


@pytest.mark.anyio
async def test_tiered_compaction_subtracts_reclaim_from_provider_usage():
    from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
    from pydantic_ai.models.test import TestModel
    from pydantic_ai.tools import RunContext
    from pydantic_ai.usage import RequestUsage, RunUsage
    from pydantic_ai_harness.compaction import SlidingWindowCompaction
    from pydantic_ai_harness.compaction._shared import get_compaction_reclaim

    from rpent.planner.compaction import ImageAwareCompaction

    history = [
        ModelRequest(parts=[UserPromptPart("task")]),
        *observation("old"),
        ModelResponse(parts=[TextPart("next")], usage=RequestUsage(input_tokens=2200)),
        ModelRequest(parts=[UserPromptPart("continue")]),
    ]

    class Summarizer:
        async def compact(self, messages, ctx):
            pytest.fail("The sliding window already reclaimed the old image tokens")

    strategy = ImageAwareCompaction(
        tiers=[
            SlidingWindowCompaction(max_messages=4, keep_messages=2, receipts=False),
            Summarizer(),
        ],
        target_tokens=1000,
    )
    model = TestModel()
    request = ModelRequestContext(
        model=model,
        messages=history,
        model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    result = await strategy.before_model_request(
        RunContext(deps=None, model=model, usage=RunUsage()), request
    )
    compacted = result.messages
    assert compacted[0] == history[0]
    assert compacted[-2:] == history[-2:]
    assert count_image_tokens_in(compacted) == 0
    assert get_compaction_reclaim(result) >= 2048
