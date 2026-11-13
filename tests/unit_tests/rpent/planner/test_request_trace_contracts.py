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


"""Historical wire trace admission remains readable after recorder migration."""

import json

import pytest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _episode_fixture(path):

    trace = path / "api_trace"
    directory = trace / "requests/000001"
    write_json(directory / "request.json", {"messages": []})
    write_json(
        directory / "metadata.json",
        {
            "complete": True,
            "status_code": 200,
            "content_type": "application/json",
            "native_id": "000001",
        },
    )
    write_json(
        directory / "response.body",
        {
            "choices": [
                {
                    "message": {"role": "assistant", "content": "done"},
                    "finish_reason": "stop",
                }
            ]
        },
    )
    write_json(
        trace / "responses/000001.json",
        [{"parts": [{"part_kind": "text", "content": "done"}]}],
    )
    write_json(
        path / "result.json",
        {"success": False, "agent_error": None, "stats": {"turns_used": 1}},
    )
    write_json(path / "transcript_test.json", {"error": None})
    write_json(path / "states.json", {"steps": [{"terminated": False}]})
    write_json(path / "rollout.json", {})
    return directory


def test_native_task_failure_is_valid_but_success_must_match_environment(tmp_path):
    from rpent.evaluation.trajectory import audit_episode

    _episode_fixture(tmp_path)
    result, turns = audit_episode(tmp_path)
    assert not result["success"] and len(turns) == 1
    path = tmp_path / "result.json"
    result = json.loads(path.read_text())
    result["success"] = True
    path.write_text(json.dumps(result))
    with pytest.raises(ValueError, match="native success mismatch"):
        audit_episode(tmp_path)


def test_wire_native_output_mismatch_cannot_enter_sft(tmp_path):
    from rpent.evaluation.trajectory import audit_episode

    _episode_fixture(tmp_path)
    path = tmp_path / "api_trace/responses/000001.json"
    value = json.loads(path.read_text())
    value[0]["parts"][0]["content"] = "different"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="completed turn coverage"):
        audit_episode(tmp_path)


@pytest.mark.parametrize("finish_reason", ["length", None])
def test_output_limit_cannot_enter_sft(tmp_path, finish_reason):
    from rpent.evaluation.trajectory import audit_episode

    directory = _episode_fixture(tmp_path)
    value = json.loads((directory / "response.body").read_text())
    value["choices"][0]["finish_reason"] = finish_reason
    (directory / "response.body").write_text(json.dumps(value))
    with pytest.raises(ValueError, match="completed turn coverage"):
        audit_episode(tmp_path)


def test_stream_reader_requires_done_and_keeps_tool_fragments(tmp_path):
    from rpent.evaluation.trajectory import read_response

    write_json(
        tmp_path / "metadata.json",
        {
            "complete": True,
            "status_code": 200,
            "content_type": "text/event-stream",
        },
    )
    chunks = [
        {
            "choices": [
                {
                    "delta": {
                        "reasoning_content": "think",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call1",
                                "function": {"name": "finish", "arguments": "{"},
                            }
                        ],
                    }
                }
            ]
        },
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"index": 0, "function": {"arguments": "}"}}]
                    },
                    "finish_reason": "tool_calls",
                }
            ]
        },
    ]
    body = "\n\n".join("data: " + json.dumps(chunk) for chunk in chunks)
    (tmp_path / "response.body").write_text(body)
    with pytest.raises(ValueError, match="SSE lacks DONE"):
        read_response(tmp_path)
    (tmp_path / "response.body").write_text(body + "\n\ndata: [DONE]\n\n")
    message, metadata = read_response(tmp_path)
    assert message["reasoning_content"] == "think"
    assert message["tool_calls"][0]["function"] == {"name": "finish", "arguments": "{}"}
    assert metadata["finish_reason"] == "tool_calls"


@pytest.mark.parametrize("finish_reason", ["stop", "length"])
def test_canonical_capture_checks_completion_and_attachments(tmp_path, finish_reason):
    from rpent.evaluation.trajectory import audit_episode
    from rpent.planner.base import PlannerResult
    from rpent.session import EnvState, Payload
    from rpent.session.planner_capture import end_planner_task, start_planner_task

    _episode_fixture(tmp_path)
    state = start_planner_task(EnvState(tmp_path), instruction="task", config={})
    state.register_planner(
        backend="api",
        status="completed",
        request=Payload(
            "pydantic_ai.model_request", {"image": b"original"}, completeness="complete"
        ),
        response=Payload(
            "pydantic_ai.model_response",
            {"finish_reason": finish_reason},
            completeness="complete",
        ),
    )
    end_planner_task(
        state,
        PlannerResult(stats={"turns_used": 1}),
        finish_result=None,
        environment_success=False,
        exception=None,
    )
    if finish_reason == "length":
        with pytest.raises(ValueError, match="finish reason"):
            audit_episode(tmp_path, verify_assets=True)
        return
    _, turns = audit_episode(tmp_path, verify_assets=True)
    assert len(turns) == 1 and turns[0][2]["capture_boundary"] == "pydantic_ai"
    # Canonical records take precedence: a missing attachment cannot fall back
    # to the complete historical wire trace also present in this fixture.
    next(state.trajectory_dir.glob("artifacts/*.bin")).unlink()
    with pytest.raises(FileNotFoundError):
        audit_episode(tmp_path, verify_assets=True)
