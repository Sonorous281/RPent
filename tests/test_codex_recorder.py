# Copyright 2026 The RPent Authors.

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import BaseModel

import rpent.planner.codex as codex_module
from robots.libero import get_env_spec as get_libero_env_spec
from robots.robotwin import get_env_spec as get_robotwin_env_spec
from robots.robotwin.toolkit import RoboTwinToolkit
from rpent.planner.codex import (
    CodexPlanner,
    _codex_mcp_config_overrides,
    _Recorder,
)
from rpent.planner.codex_runtime import inspect_runtime
from rpent.tools.toolkit import Toolkit


def test_recorder_tracks_logical_and_sdk_turn_for_mcp_calls():
    recorder = _Recorder(max_turns=10)
    recorder.observe(
        {
            "method": "item/completed",
            "payload": {
                "turn_id": "sdk-turn-1",
                "item": {
                    "type": "agentMessage",
                    "text": "I will inspect the state.",
                },
            },
        }
    )

    recorder.observe(
        {
            "method": "item/completed",
            "payload": {
                "turn_id": "sdk-turn-1",
                "item": {
                    "type": "mcpToolCall",
                    "id": "sdk-call-1",
                    "tool": "view_driver_state",
                },
            },
        }
    )

    assert recorder.stats()["mcp_tool_events"] == [
        {
            "sequence": 1,
            "turn_id": "turn-000001",
            "sdk_turn_id": "sdk-turn-1",
            "sdk_tool_call_id": "sdk-call-1",
            "tool": "view_driver_state",
        }
    ]


def test_codex_config_override_pins_reasoning_effort():
    overrides = _codex_mcp_config_overrides(
        mcp_url="http://127.0.0.1:1234/mcp/",
        base_url=None,
        reasoning_effort="high",
    )

    assert 'model_reasoning_effort="high"' in overrides


def test_inspect_runtime_records_sdk_and_binary(monkeypatch, tmp_path):
    binary = tmp_path / "codex"
    binary.write_text("")
    monkeypatch.setattr(
        "rpent.planner.codex_runtime.subprocess.run",
        Mock(
            return_value=Mock(
                stdout="codex-cli 0.144.1\n",
                stderr="",
            )
        ),
    )
    monkeypatch.setattr(
        "rpent.planner.codex_runtime.importlib.metadata.version",
        Mock(return_value="0.1.0b3"),
    )

    runtime = inspect_runtime(
        model="gpt-5.5",
        reasoning_effort="high",
        base_url="https://example.invalid/v1",
        configured_binary=str(binary),
    )

    assert runtime["model"] == "gpt-5.5"
    assert runtime["reasoning_effort"] == "high"
    assert runtime["sdk_version"] == "0.1.0b3"
    assert runtime["binary_source"] == "configured"
    assert runtime["binary_version"] == "0.144.1"
    assert runtime["provider"] == "rpent_proxy"


def test_codex_planner_writes_runtime_manifest(monkeypatch, tmp_path):
    runtime = {
        "model": "gpt-5.5",
        "reasoning_effort": "high",
        "sdk_version": "0.1.0b3",
        "binary_path": "/usr/bin/codex",
        "binary_resolved_path": "/usr/bin/codex",
        "binary_version": "0.144.1",
        "provider": "rpent_proxy",
        "prompt_sha256": None,
        "tool_schema_sha256": None,
    }
    monkeypatch.setattr(
        "rpent.planner.codex.inspect_runtime",
        Mock(return_value=runtime),
    )

    CodexPlanner(
        output_dir=str(tmp_path),
        repo_root=tmp_path,
        model="gpt-5.5",
        reasoning_effort="high",
    )

    manifest = json.loads((tmp_path / "codex_runtime_manifest.json").read_text())
    assert manifest == runtime


def _complete_response(recorder, *items):
    for item in items:
        recorder.observe(
            {
                "method": "item/completed",
                "payload": {"item": item},
            }
        )
    recorder.observe(
        {
            "method": "thread/tokenUsage/updated",
            "payload": {"token_usage": {}},
        }
    )
    recorder.observe(
        {
            "method": "item/started",
            "payload": {"item": {"type": "reasoning", "id": "next-response"}},
        }
    )


class _SdkMcpContent(BaseModel):
    type: str = "text"
    text: str


class _SdkMcpResult(BaseModel):
    content: list[_SdkMcpContent]


class _SdkMcpItem(BaseModel):
    type: str = "mcpToolCall"
    server: str = "rpent"
    tool: str
    status: str = "completed"
    id: str
    arguments: dict = {}
    result: _SdkMcpResult


def _sdk_mcp_item(
    tool: str,
    *,
    frame_id: int,
    success: bool = False,
    state_trustworthy: bool = True,
) -> _SdkMcpItem:
    payload = {
        "frame_id": frame_id,
        "step_idx": frame_id,
        "episode_status": {
            "mutation_seq": frame_id,
            "take_action_cnt": frame_id,
            "success": success,
            "state_trustworthy": state_trustworthy,
        },
    }
    if tool == "finish":
        payload["_finish"] = True
    return _SdkMcpItem(
        tool=tool,
        id=f"call-{tool}-{frame_id}",
        result=_SdkMcpResult(
            content=[_SdkMcpContent(text=json.dumps(payload))]
        ),
    )


def _robotwin_recorder() -> _Recorder:
    toolkit = RoboTwinToolkit.__new__(RoboTwinToolkit)
    return _Recorder(
        max_turns=10,
        enforce_action_guard=True,
        progress_evaluator=toolkit.evaluate_progress,
    )


def test_sdk_pydantic_progress_matches_dict_progress():
    toolkit = RoboTwinToolkit.__new__(RoboTwinToolkit)
    sdk_item = _sdk_mcp_item("move_to", frame_id=1)
    dict_signal = toolkit.evaluate_progress(
        sdk_item.model_dump(mode="json", exclude_none=True)
    )
    recorder = _robotwin_recorder()

    _complete_response(recorder, sdk_item)

    assert dict_signal is not None
    assert recorder.no_action_responses == 0
    assert recorder.stats()["planner_runtime_progress_adapter_error"] == 0


def test_five_distinct_sdk_progress_results_do_not_steer_or_abort():
    recorder = _robotwin_recorder()

    for frame_id, tool in enumerate(
        (
            "view_driver_state",
            "sample_world_xyz",
            "query_world_map",
            "sample_world_xyz",
            "move_to",
        ),
        start=1,
    ):
        _complete_response(
            recorder,
            _sdk_mcp_item(tool, frame_id=frame_id),
        )

    assert recorder.consume_guard_warning() is None
    assert recorder.no_action_responses == 0
    assert recorder.abort_requested is False


def test_sdk_finish_object_latches_terminal():
    recorder = _robotwin_recorder()

    _complete_response(
        recorder,
        _sdk_mcp_item(
            "lingbot_act",
            frame_id=1,
            success=True,
        ),
    )
    recorder.consume_guard_warning()
    recorder.observe(
        {
            "method": "item/started",
            "payload": {
                "item": _sdk_mcp_item(
                    "finish",
                    frame_id=2,
                    success=True,
                )
            },
        }
    )
    recorder.observe(
        {
            "method": "item/completed",
            "payload": {
                "item": _sdk_mcp_item(
                    "finish",
                    frame_id=2,
                    success=True,
                )
            },
        }
    )

    assert recorder.stats()["terminal_latched"] is True
    assert recorder.abort_requested is False


def test_progress_adapter_failure_is_runtime_error_not_no_progress():
    recorder = _Recorder(
        max_turns=10,
        enforce_action_guard=True,
        progress_evaluator=lambda _item: (_ for _ in ()).throw(
            ValueError("bad progress payload")
        ),
    )

    _complete_response(
        recorder,
        _sdk_mcp_item("move_to", frame_id=1),
    )

    assert recorder.progress_adapter_error is True
    assert recorder.abort_requested is True
    assert recorder.error == (
        "planner_runtime_progress_adapter_error: "
        "ValueError: bad progress payload"
    )
    assert recorder.stats()["planner_runtime_progress_adapter_error"] == 1
    assert recorder.no_action_responses == 0


def test_action_guard_is_disabled_by_default():
    recorder = _Recorder(max_turns=10)

    for _ in range(6):
        _complete_response(recorder, {"type": "reasoning"})

    assert recorder.abort_requested is False
    assert recorder.consume_guard_warning() is None
    assert recorder.stats()["max_consecutive_no_action_responses"] == 0


def test_only_robotwin_env_spec_enables_planner_progress_guard():
    assert get_robotwin_env_spec().planner_progress_guard is True
    assert get_libero_env_spec().planner_progress_guard is False


def test_action_guard_warns_at_three_no_action_responses():
    recorder = _Recorder(max_turns=10, enforce_action_guard=True)

    for _ in range(3):
        _complete_response(recorder, {"type": "agentMessage", "text": "Calling now."})

    assert recorder.abort_requested is False
    assert "Call exactly one registered environment tool" in (
        recorder.consume_guard_warning() or ""
    )
    assert recorder.stats()["planner_guard_warning_count"] == 1


def test_action_guard_aborts_at_five_no_action_responses():
    recorder = _Recorder(max_turns=10, enforce_action_guard=True)

    for _ in range(5):
        _complete_response(recorder, {"type": "reasoning"})

    stats = recorder.stats()
    assert recorder.abort_requested is True
    assert recorder.error is not None
    assert recorder.error.startswith("planner_no_action_loop:")
    assert stats["max_consecutive_no_action_responses"] == 5
    assert stats["planner_no_action_loop"] == 1


def test_repeated_identical_rpent_result_does_not_reset_progress_streak():
    recorder = _Recorder(
        max_turns=10,
        enforce_action_guard=True,
        progress_evaluator=_robotwin_progress,
    )
    item = {
        "type": "mcpToolCall",
        "server": "rpent",
        "tool": "view_driver_state",
        "arguments": {"step": 0},
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "frame_id": 0,
                            "step_idx": 0,
                            "episode_status": {
                                "mutation_seq": 0,
                                "take_action_cnt": 0,
                            },
                        }
                    ),
                }
            ]
        },
    }

    _complete_response(recorder, item)
    _complete_response(recorder, item)

    assert recorder.no_tool_call_responses == 0
    assert recorder.no_action_responses == 1


def test_changed_mutation_marker_counts_as_new_environment_progress():
    recorder = _Recorder(
        max_turns=10,
        enforce_action_guard=True,
        progress_evaluator=_robotwin_progress,
    )

    for mutation_seq in (0, 1):
        _complete_response(
            recorder,
            {
                "type": "mcpToolCall",
                "server": "rpent",
                "tool": "view_driver_state",
                "arguments": {"step": None},
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(
                                {
                                    "frame_id": mutation_seq,
                                    "episode_status": {
                                        "mutation_seq": mutation_seq,
                                        "take_action_cnt": mutation_seq,
                                    },
                                }
                            ),
                        }
                    ]
                },
            },
        )

    assert recorder.no_action_responses == 0


def _native_status_tool_item(
    *,
    success: bool,
    state_trustworthy: bool,
    tool: str = "lingbot_act",
):
    result_payload = {
        "episode_status": {
            "success": success,
            "state_trustworthy": state_trustworthy,
            "mutation_seq": 1,
        }
    }
    if tool == "finish":
        result_payload["_finish"] = True
    return {
        "type": "mcpToolCall",
        "server": "rpent",
        "tool": tool,
        "status": "completed",
        "arguments": {},
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(result_payload),
                }
            ]
        },
    }


def _robotwin_progress(item):
    result = item.get("result", {}) if isinstance(item, dict) else {}
    content = result.get("content", []) if isinstance(result, dict) else []
    text = content[0].get("text", "") if content else ""
    payload = json.loads(text) if text else {}
    status = payload.get("episode_status", {})
    token = {
        key: status[key]
        for key in ("mutation_seq", "take_action_cnt")
        if key in status
    }
    return (
        {
            "requires_finish": True,
            "environment_progress": True,
            "progress_token": token,
        }
        if '"success": true' in text and '"state_trustworthy": true' in text
        else ({"progress_token": token} if token else None)
    )


def test_native_success_requires_real_finish_on_next_response():
    recorder = _Recorder(
        max_turns=10,
        enforce_action_guard=True,
        progress_evaluator=_robotwin_progress,
    )

    _complete_response(
        recorder,
        _native_status_tool_item(success=True, state_trustworthy=True),
    )

    warning = recorder.consume_guard_warning() or ""
    assert "next model response" in warning
    assert "required terminal tool" in warning
    assert recorder.abort_requested is False

    _complete_response(
        recorder,
        {
            **_native_status_tool_item(
                success=True,
                state_trustworthy=True,
                tool="finish",
            ),
            "arguments": {"status": "success", "summary": "done"},
        },
    )

    assert recorder.abort_requested is False
    assert recorder.finish_result == {
        "_finish": True,
        "status": "success",
        "summary": "done",
    }
    assert recorder.stats()["native_success_without_finish"] == 0


def test_successful_finish_latches_terminal_before_trailing_sdk_events():
    recorder = _Recorder(
        max_turns=10,
        enforce_action_guard=True,
        progress_evaluator=_robotwin_progress,
    )

    _complete_response(
        recorder,
        _native_status_tool_item(success=True, state_trustworthy=True),
    )
    recorder.observe(
        {
            "method": "item/started",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "server": "rpent",
                    "tool": "finish",
                    "status": "inProgress",
                }
            },
        }
    )
    finish_item = {
        **_native_status_tool_item(
            success=True,
            state_trustworthy=True,
            tool="finish",
        ),
        "arguments": {"status": "success", "summary": "done"},
    }
    recorder.observe(
        {
            "method": "item/completed",
            "payload": {"item": finish_item},
        }
    )
    stats_after_finish = recorder.stats()

    for _ in range(6):
        recorder.observe(
            {
                "method": "thread/tokenUsage/updated",
                "payload": {"token_usage": {}},
            }
        )
        recorder.observe(
            {
                "method": "item/started",
                "payload": {"item": {"type": "reasoning"}},
            }
        )
        recorder.observe(
            {
                "method": "item/completed",
                "payload": {"item": {"type": "reasoning"}},
            }
        )

    stats = recorder.stats()
    assert stats["terminal_latched"] is True
    assert stats["guard_state"] == "terminated"
    assert stats["planner_no_action_loop"] == 0
    assert stats["max_consecutive_no_action_responses"] == (
        stats_after_finish["max_consecutive_no_action_responses"]
    )
    assert recorder.consume_guard_warning() is None
    assert recorder.error is None


def test_token_usage_is_not_itself_a_guard_response():
    recorder = _Recorder(max_turns=10, enforce_action_guard=True)
    recorder.observe(
        {
            "method": "item/completed",
            "payload": {"item": {"type": "reasoning"}},
        }
    )

    for _ in range(10):
        recorder.observe(
            {
                "method": "thread/tokenUsage/updated",
                "payload": {"token_usage": {}},
            }
        )

    assert recorder.stats()["logical_response_count"] == 0
    assert recorder.no_action_responses == 0

    recorder.observe(
        {
            "method": "item/started",
            "payload": {"item": {"type": "reasoning"}},
        }
    )
    assert recorder.stats()["logical_response_count"] == 1
    assert recorder.no_action_responses == 1


def test_failed_finish_result_does_not_latch_terminal():
    recorder = _Recorder(
        max_turns=10,
        enforce_action_guard=True,
        progress_evaluator=_robotwin_progress,
    )
    recorder.observe(
        {
            "method": "item/started",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "server": "rpent",
                    "tool": "finish",
                    "status": "inProgress",
                }
            },
        }
    )
    recorder.observe(
        {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "mcpToolCall",
                    "server": "rpent",
                    "tool": "finish",
                    "status": "failed",
                    "arguments": {"status": "success"},
                    "result": {"content": [{"type": "text", "text": "{}"}]},
                }
            },
        }
    )

    stats = recorder.stats()
    assert stats["terminal_latched"] is False
    assert stats["terminal_tool_failure"] == 1
    assert recorder.finish_result is None
    assert recorder.abort_requested is True
    assert recorder.error is not None
    assert recorder.error.startswith("planner_terminal_tool_failed:")


def test_native_success_text_only_response_aborts_with_specific_reason():
    recorder = _Recorder(
        max_turns=10,
        enforce_action_guard=True,
        progress_evaluator=_robotwin_progress,
    )

    _complete_response(
        recorder,
        _native_status_tool_item(success=True, state_trustworthy=True),
    )
    _complete_response(
        recorder,
        {"type": "agentMessage", "text": "The task is complete."},
    )

    assert recorder.abort_requested is True
    assert recorder.error is not None
    assert "native_success_without_finish" in recorder.error
    assert recorder.stats()["planner_no_action_loop"] == 1
    assert recorder.stats()["native_success_without_finish"] == 1


@pytest.mark.parametrize(
    ("success", "state_trustworthy"),
    [(False, True), (True, False)],
)
def test_stale_or_untrusted_status_does_not_trigger_finish_only_guard(
    success,
    state_trustworthy,
):
    recorder = _Recorder(
        max_turns=10,
        enforce_action_guard=True,
        progress_evaluator=_robotwin_progress,
    )

    _complete_response(
        recorder,
        _native_status_tool_item(
            success=success,
            state_trustworthy=state_trustworthy,
        ),
    )
    _complete_response(recorder, {"type": "reasoning"})

    assert recorder.abort_requested is False
    assert recorder.stats()["native_success_finish_required_count"] == 0
    assert recorder.stats()["native_success_without_finish"] == 0


def test_repeated_image_view_does_not_count_as_new_environment_progress():
    recorder = _Recorder(max_turns=10, enforce_action_guard=True)

    _complete_response(
        recorder,
        {"type": "imageView", "path": "/tmp/frame-1.png"},
    )
    _complete_response(
        recorder,
        {"type": "imageView", "path": "/tmp/frame-1.png"},
    )

    assert recorder.no_tool_call_responses == 0
    assert recorder.no_action_responses == 1
    assert (
        recorder.stats()["max_consecutive_no_environment_progress_responses"]
        == 1
    )


def test_action_guard_thresholds_are_configurable():
    recorder = _Recorder(
        max_turns=10,
        enforce_action_guard=True,
        action_guard_warn_after=2,
        action_guard_abort_after=4,
    )

    for _ in range(2):
        _complete_response(recorder, {"type": "reasoning"})
    assert recorder.abort_requested is False
    assert recorder.consume_guard_warning() is not None

    for _ in range(2):
        _complete_response(recorder, {"type": "reasoning"})
    assert recorder.abort_requested is True
    assert recorder.stats()["planner_action_guard"] == {
        "enabled": True,
        "warn_after": 2,
        "abort_after": 4,
    }


def test_codex_session_guard_steers_interrupts_and_cleans_up(tmp_path, monkeypatch):
    class FakeTurn:
        def __init__(self):
            self.steered = []
            self.interrupt_count = 0

        def stream(self):
            for index in range(5):
                yield {
                    "method": "item/started",
                    "payload": {
                        "turn_id": "sdk-turn-1",
                        "item": {
                            "id": f"reasoning-{index}",
                            "type": "reasoning",
                        },
                    },
                }
                yield {
                    "method": "item/completed",
                    "payload": {
                        "turn_id": "sdk-turn-1",
                        "item": {
                            "type": "reasoning",
                            "summary": f"idle response {index}",
                        },
                    },
                }
                yield {
                    "method": "thread/tokenUsage/updated",
                    "payload": {"token_usage": {}},
                }
            yield {
                "method": "turn/completed",
                "payload": {"turn": {"id": "turn-1", "status": "interrupted"}},
            }

        def steer(self, warning):
            self.steered.append(warning)

        def interrupt(self):
            self.interrupt_count += 1

    class FakeThread:
        def __init__(self, turn):
            self.fake_turn = turn

        def turn(self, *_args, **_kwargs):
            return self.fake_turn

    class FakeCodex:
        exited = False

        def __init__(self, *_args, **_kwargs):
            self.turn = FakeTurn()

        def __enter__(self):
            fake_runtime["codex"] = self
            return self

        def __exit__(self, *_args):
            type(self).exited = True

        def thread_start(self, **_kwargs):
            return FakeThread(self.turn)

    class FakeMcpServer:
        stopped = False

        def __init__(self, _toolkit):
            pass

        def start(self):
            return "http://127.0.0.1:1/mcp"

        def stop(self):
            type(self).stopped = True

    fake_runtime = {}
    monkeypatch.setattr(codex_module.openai_codex, "Codex", FakeCodex)
    monkeypatch.setattr(codex_module, "HttpMcpServer", FakeMcpServer)
    monkeypatch.setattr(
        CodexPlanner,
        "_build_config",
        lambda _self, _mcp_url: object(),
    )

    planner = object.__new__(CodexPlanner)
    planner._output_dir = str(tmp_path)
    planner._repo_root = str(tmp_path)
    planner._timeout_s = 5
    planner._extra_dirs = []
    planner._output_path = tmp_path / "codex.out"
    planner._model = "diagnostic-model"
    planner._reasoning_effort = "high"
    planner._enforce_action_guard = True
    planner._action_guard_warn_after = 3
    planner._action_guard_abort_after = 5
    planner._dashboard = None
    planner._runtime_manifest_path = tmp_path / "codex_runtime_manifest.json"
    planner._runtime_manifest = {}

    result = planner.solve(
        system_prompt="diagnostic",
        user_message="emit no tools",
        toolkit=Toolkit(),
        max_turns=10,
    )

    fake_codex = fake_runtime["codex"]
    assert len(fake_codex.turn.steered) == 1
    assert fake_codex.turn.interrupt_count == 1
    assert FakeCodex.exited is True
    assert FakeMcpServer.stopped is True
    assert result.error is not None
    assert result.error.startswith("planner_no_action_loop:")
    assert result.stats["planner_no_action_loop"] == 1
    assert result.stats["max_consecutive_no_environment_progress_responses"] == 5
    assert result.stats["interrupt_requested"] is True
    assert result.stats["interrupt_origin"] == "guard_abort"
    assert result.stats["interrupt_count"] == 1
    assert result.stats["interrupt_acknowledged"] is True
    assert Path(result.stats["raw_stream_path"]).is_file()


def test_codex_interrupt_without_terminal_event_forces_bounded_close(
    tmp_path,
    monkeypatch,
):
    import threading

    release_stream = threading.Event()

    class FakeTurn:
        def __init__(self):
            self.interrupt_count = 0

        def stream(self):
            for index in range(5):
                yield {
                    "method": "item/completed",
                    "payload": {"item": {"type": "reasoning"}},
                }
                yield {
                    "method": "thread/tokenUsage/updated",
                    "payload": {"token_usage": {}},
                }
                yield {
                    "method": "item/started",
                    "payload": {"item": {"type": "reasoning"}},
                }
            release_stream.wait()

        def steer(self, _warning):
            pass

        def interrupt(self):
            self.interrupt_count += 1

    class FakeThread:
        def __init__(self, turn):
            self.fake_turn = turn

        def turn(self, *_args, **_kwargs):
            return self.fake_turn

    class FakeCodex:
        close_count = 0

        def __init__(self, *_args, **_kwargs):
            self.turn = FakeTurn()

        def __enter__(self):
            fake_runtime["codex"] = self
            return self

        def __exit__(self, *_args):
            self.close()

        def close(self):
            type(self).close_count += 1
            release_stream.set()

        def thread_start(self, **_kwargs):
            return FakeThread(self.turn)

    class FakeMcpServer:
        stopped = False

        def __init__(self, _toolkit):
            pass

        def start(self):
            return "http://127.0.0.1:1/mcp"

        def stop(self):
            type(self).stopped = True

    fake_runtime = {}
    monkeypatch.setattr(codex_module.openai_codex, "Codex", FakeCodex)
    monkeypatch.setattr(codex_module, "HttpMcpServer", FakeMcpServer)
    monkeypatch.setattr(
        CodexPlanner,
        "_build_config",
        lambda _self, _mcp_url: object(),
    )

    planner = object.__new__(CodexPlanner)
    planner._output_dir = str(tmp_path)
    planner._repo_root = str(tmp_path)
    planner._timeout_s = 3600
    planner._interrupt_grace_s = 0.05
    planner._extra_dirs = []
    planner._output_path = tmp_path / "codex.out"
    planner._model = "diagnostic-model"
    planner._reasoning_effort = "high"
    planner._enforce_action_guard = True
    planner._action_guard_warn_after = 3
    planner._action_guard_abort_after = 5
    planner._dashboard = None
    planner._runtime_manifest_path = tmp_path / "codex_runtime_manifest.json"
    planner._runtime_manifest = {}

    result = planner.solve(
        system_prompt="diagnostic",
        user_message="emit no tools",
        toolkit=Toolkit(),
        max_turns=10,
    )

    fake_codex = fake_runtime["codex"]
    assert fake_codex.turn.interrupt_count == 1
    assert FakeCodex.close_count >= 1
    assert FakeMcpServer.stopped is True
    assert result.error == (
        "planner_runtime_failure: sdk_interrupt_did_not_converge"
    )
    assert result.stats["interrupt_count"] == 1
    assert result.stats["stream_terminal_event_seen"] is False
    assert result.stats["sdk_interrupt_did_not_converge"] == 1
    assert result.stats["elapsed_s"] < 10


def test_codex_interrupt_with_terminal_event_exits_normally(
    tmp_path,
    monkeypatch,
):
    class FakeTurn:
        def __init__(self):
            self.interrupt_count = 0

        def stream(self):
            for _ in range(5):
                yield {
                    "method": "item/completed",
                    "payload": {"item": {"type": "reasoning"}},
                }
                yield {
                    "method": "thread/tokenUsage/updated",
                    "payload": {"token_usage": {}},
                }
                yield {
                    "method": "item/started",
                    "payload": {"item": {"type": "reasoning"}},
                }
            yield {
                "method": "turn/completed",
                "payload": {"turn": {"id": "turn-1", "status": "interrupted"}},
            }

        def steer(self, _warning):
            pass

        def interrupt(self):
            self.interrupt_count += 1

    class FakeThread:
        def __init__(self, turn):
            self.fake_turn = turn

        def turn(self, *_args, **_kwargs):
            return self.fake_turn

    class FakeCodex:
        def __init__(self, *_args, **_kwargs):
            self.turn = FakeTurn()

        def __enter__(self):
            fake_runtime["codex"] = self
            return self

        def __exit__(self, *_args):
            pass

        def thread_start(self, **_kwargs):
            return FakeThread(self.turn)

    class FakeMcpServer:
        stopped = False

        def __init__(self, _toolkit):
            pass

        def start(self):
            return "http://127.0.0.1:1/mcp"

        def stop(self):
            type(self).stopped = True

    fake_runtime = {}
    monkeypatch.setattr(codex_module.openai_codex, "Codex", FakeCodex)
    monkeypatch.setattr(codex_module, "HttpMcpServer", FakeMcpServer)
    monkeypatch.setattr(
        CodexPlanner,
        "_build_config",
        lambda _self, _mcp_url: object(),
    )

    planner = object.__new__(CodexPlanner)
    planner._output_dir = str(tmp_path)
    planner._repo_root = str(tmp_path)
    planner._timeout_s = 3600
    planner._interrupt_grace_s = 0.05
    planner._extra_dirs = []
    planner._output_path = tmp_path / "codex.out"
    planner._model = "diagnostic-model"
    planner._reasoning_effort = "high"
    planner._enforce_action_guard = True
    planner._action_guard_warn_after = 3
    planner._action_guard_abort_after = 5
    planner._dashboard = None
    planner._runtime_manifest_path = tmp_path / "codex_runtime_manifest.json"
    planner._runtime_manifest = {}

    result = planner.solve(
        system_prompt="diagnostic",
        user_message="emit no tools",
        toolkit=Toolkit(),
        max_turns=10,
    )

    fake_codex = fake_runtime["codex"]
    assert fake_codex.turn.interrupt_count == 1
    assert FakeMcpServer.stopped is True
    assert result.stats["interrupt_count"] == 1
    assert result.stats["stream_terminal_event_seen"] is True
    assert result.stats["sdk_interrupt_did_not_converge"] == 0
    assert result.error is not None
    assert result.error.startswith("planner_no_action_loop:")
