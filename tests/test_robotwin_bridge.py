# Copyright 2026 The RPent Authors.

import argparse
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from robots.robotwin import (
    CHECKPOINT_REVISION,
    _parse_config,
    _resolve_cuda_devices,
    _resolve_hybrid_context,
)
from robots.robotwin.env_client import RoboTwinEnvClient
from robots.robotwin.primitives import RoboTwinPrimitives
from robots.robotwin.prompt_bundle import system_prompt, user_prompt
from robots.robotwin.telemetry import audit_codex_control_path
from robots.robotwin.toolkit import RoboTwinToolkit
from robots.robotwin.tools import (
    TOOLS_SPEC,
    dump_untrusted_status,
)
from rpent.cli.main import _resolve_finish_result
from rpent.tools.toolkit import Toolkit
from rpent.utils.rpc import RpcError

CAPABILITIES = {
    "contract_version": "robotwin-agent-v1",
    "profile": "downloads_hybrid",
    "compatibility_id": "robotwin-rpent-downloads-2026-07-31-v1",
    "server_instance_id": "server-1",
    "mutation_id_prefix": "server-1:",
}


def test_robotwin_eef_config_matches_checkpoint_feature_contract():
    config_path = (
        Path(__file__).parents[1]
        / "robots"
        / "robotwin"
        / "configs"
        / "robot_configs"
        / "robotwin_eef.yaml"
    )
    config = yaml.safe_load(config_path.read_text())

    state_arm = config["states"][0]["observation.state.arm.position"]["origin_keys"]
    state_gripper = config["states"][1]["observation.state.effector.position"][
        "origin_keys"
    ]
    action_arm = config["actions"][0]["action.arm.position"]["origin_keys"]
    action_gripper = config["actions"][1]["action.effector.position"]["origin_keys"]

    assert state_arm == [
        {"observation.state": {"start": 0, "end": 7}},
        {"observation.state": {"start": 8, "end": 15}},
    ]
    assert state_gripper == [
        {"observation.state": {"start": 7, "end": 8}},
        {"observation.state": {"start": 15, "end": 16}},
    ]
    assert action_arm == [
        {"action": {"start": 0, "end": 7}},
        {"action": {"start": 8, "end": 15}},
    ]
    assert action_gripper == [
        {"action": {"start": 7, "end": 8}},
        {"action": {"start": 15, "end": 16}},
    ]
    image_sources = [
        next(iter(item.values()))["origin_keys"] for item in config["images"]
    ]
    assert image_sources == [
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
    ]


def completed(result, *, executed_actions=0):
    return {
        "mutation_id": "mutation",
        "state": "completed",
        "executed_actions": executed_actions,
        "result": result,
    }


class TimeoutThenRecoverRpc:
    def __init__(self):
        self.calls = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append((method, kwargs))
        if method == "env.get_capabilities":
            return getattr(self, "capabilities", CAPABILITIES)
        if method == "env.execute_actions":
            raise TimeoutError("reply lost")
        if method == "env.get_mutation_result":
            return completed({"executed_actions": 2}, executed_actions=2)
        raise AssertionError(method)


def test_mutation_timeout_queries_original_id_without_resubmitting():
    rpc = TimeoutThenRecoverRpc()
    client = RoboTwinEnvClient(rpc, expected_contract=CAPABILITIES)
    observed = []
    client.mutation_observer = observed.append

    result = client.execute_actions(
        "ee",
        np.zeros((2, 16)),
        mutation_id="episode-1-chunk-1",
        expected_version={"episode_generation": 1, "mutation_seq": 0},
    )

    methods = [method for method, _ in rpc.calls]
    assert methods.count("env.execute_actions") == 1
    assert methods.count("env.get_mutation_result") == 1
    assert rpc.calls[-1][1]["mutation_id"] == "episode-1-chunk-1"
    assert result["recovered_after_transport_error"] is True
    assert [event["phase"] for event in observed] == ["submitted", "recovered"]
    assert {event["mutation_id"] for event in observed} == {"episode-1-chunk-1"}


class BusinessErrorRpc:
    def __init__(self):
        self.calls = []

    def call(self, method, args=(), kwargs=None, *, timeout_s=None):
        self.calls.append(method)
        if method == "env.get_capabilities":
            return CAPABILITIES
        raise RpcError(method, "idempotency_conflict", traceback="remote traceback")


def test_mutation_business_error_is_not_misclassified_as_transport_timeout():
    rpc = BusinessErrorRpc()
    client = RoboTwinEnvClient(rpc, expected_contract=CAPABILITIES)

    with pytest.raises(RpcError, match="idempotency_conflict"):
        client.execute_actions(
            "ee",
            np.zeros((1, 16)),
            mutation_id="conflicting-id",
            expected_version=None,
        )

    assert rpc.calls == ["env.get_capabilities", "env.execute_actions"]


def test_mutation_result_rejects_ids_from_another_server_instance():
    class MutationResultRpc:
        def call(self, method, args=(), kwargs=None, *, timeout_s=None):
            if method == "env.get_capabilities":
                return CAPABILITIES
            if method == "env.get_mutation_result":
                raise RpcError(method, "stale_server_instance")
            raise AssertionError(method)

    client = RoboTwinEnvClient(
        MutationResultRpc(),
        expected_contract=CAPABILITIES,
    )

    with pytest.raises(RpcError, match="stale_server_instance"):
        client.get_mutation_result("server-1:old-mutation")


class FakeEnv:
    def __init__(self):
        self.mutation_index = 0
        self.qpos_updates = []
        self.action_calls = []
        self.reset_calls = []
        self.success = False
        self.state = {
            "left_eef_pose": np.array([0, 0, 0, 1, 0, 0, 0], dtype=float),
            "right_eef_pose": np.array([1, 1, 1, 1, 0, 0, 0], dtype=float),
            "left_gripper": 0.2,
            "right_gripper": 0.8,
            "qpos14": np.zeros(14),
            "episode_generation": 2,
            "mutation_seq": 3,
        }

    def new_mutation_id(self, primitive_id, operation):
        self.mutation_index += 1
        return f"{primitive_id}:{operation}:{self.mutation_index}"

    def get_robot_state(self):
        return self.state

    def get_episode_status(self):
        return {
            "success": self.success,
            "terminated": self.success,
            "truncated": False,
            "take_action_cnt": 0,
            "step_lim": 450,
            "state_trustworthy": True,
        }

    def plan_arm_path(self, arm, target_pose):
        return {
            "status": "Success",
            "position": np.arange(60 * 6, dtype=float).reshape(60, 6),
            "episode_generation": 2,
            "mutation_seq": 3,
            "planner_seq": 1,
        }

    def execute_qpos_updates(self, updates, *, mutation_id, expected_plan):
        self.qpos_updates.append((updates, mutation_id, expected_plan))
        return completed(
            {
                "executed_actions": len(updates),
                "episode_status": self.get_episode_status(),
            },
            executed_actions=len(updates),
        )

    def capture_policy_observation(self):
        return {
            "images": {
                "cam_high": np.zeros((2, 2, 3), dtype=np.uint8),
                "cam_left_wrist": np.zeros((2, 2, 3), dtype=np.uint8),
                "cam_right_wrist": np.zeros((2, 2, 3), dtype=np.uint8),
            },
            "state": np.zeros(16),
            "task": "native instruction",
            "episode_generation": 2,
            "mutation_seq": 3,
        }

    def execute_actions(
        self,
        action_type,
        actions,
        *,
        mutation_id,
        expected_version,
    ):
        self.action_calls.append(
            (action_type, actions.copy(), mutation_id, expected_version)
        )
        return completed(
            {
                "executed_actions": len(actions),
                "episode_status": self.get_episode_status(),
            },
            executed_actions=len(actions),
        )

    def reset_episode(self, seed, *, mutation_id, reset_options):
        self.reset_calls.append((seed, mutation_id, reset_options))
        return completed(
            {
                "requested_seed": seed,
                "actual_seed": seed,
                "instruction": "native instruction",
                "episode_status": self.get_episode_status(),
            }
        )


class FakeModel:
    def __init__(self):
        self.observations = []

    def infer(self, observation):
        self.observations.append(observation)
        return np.zeros((80, 16))


def make_primitives():
    return RoboTwinPrimitives(
        env=FakeEnv(),
        model=FakeModel(),
        seed=100002,
    )


def test_move_to_matches_downloads_25_point_downsampling():
    primitives = make_primitives()
    result = primitives.move_to(arm="left", xyz=[0.1, 0.2, 0.3])

    updates, _, expected = primitives.env.qpos_updates[-1]
    indices = np.linspace(0, 59, 25).astype(int)
    assert len(updates) == 25
    np.testing.assert_array_equal(
        np.stack([update["arm_qpos"] for update in updates]),
        np.arange(60 * 6, dtype=float).reshape(60, 6)[indices],
    )
    assert expected == {"episode_generation": 2, "mutation_seq": 3}
    assert result["executed_steps"] == 25


def test_lingbot_uses_native_prompt_and_limits_chunk_to_50():
    primitives = make_primitives()
    result = primitives.lingbot_act(chunks=1, prompt="agent rewrite")

    action_type, actions, _, version = primitives.env.action_calls[-1]
    assert action_type == "ee"
    assert actions.shape == (50, 16)
    assert primitives.model.observations[-1]["task"] == "native instruction"
    assert version == {"episode_generation": 2, "mutation_seq": 3}
    assert result["agent_prompt_ignored"] is True


def test_partial_mutation_preserves_native_action_count():
    primitives = make_primitives()
    primitives.env.execute_qpos_updates = lambda *args, **kwargs: {
        "mutation_id": "partial",
        "state": "partially_applied",
        "executed_actions": 4,
        "episode_status": primitives.env.get_episode_status(),
        "result": None,
        "error": "RuntimeError: native failure",
    }

    result = primitives.set_gripper(arm="left", val=1.0)

    assert result["success"] is False
    assert result["executed_steps"] == 4
    assert primitives.native_actions == 4


def test_finish_never_promotes_requested_success_over_native_status():
    primitives = make_primitives()
    result = primitives.finish(status="success", summary="looks done")

    assert result["_finish"] is True
    assert result["requested_success"] is True
    assert result["success"] is False
    assert result["status"] == "failure"
    assert result["native_status"]["success"] is False


def test_finish_rejects_native_success_when_episode_state_is_untrusted():
    primitives = make_primitives()
    primitives.env.success = True
    original_status = primitives.env.get_episode_status

    def untrusted_status():
        return {
            **original_status(),
            "state_trustworthy": False,
            "termination_reason": "state_unknown",
        }

    primitives.env.get_episode_status = untrusted_status
    result = primitives.finish(status="success", summary="looks done")

    assert result["_finish"] is True
    assert result["requested_success"] is True
    assert result["success"] is False
    assert result["state_trustworthy"] is False
    assert result["status"] == "failure"
    assert result["native_status"]["success"] is True


def test_reset_forwards_formal_downloads_seed_and_instruction_contract():
    env = FakeEnv()
    primitives = RoboTwinPrimitives(
        env=env,
        model=FakeModel(),
        seed=100123,
        seed_mode="exact",
        allow_infeasible=True,
        instruction_type="unseen",
        instruction="place the fan on the marked area",
    )

    result = primitives.reset()

    assert result["success"] is True
    seed, _, options = env.reset_calls[-1]
    assert seed == 100123
    assert options == {
        "exact_seed": True,
        "allow_infeasible": True,
        "instruction_type": "unseen",
        "feasibility_precheck": True,
        "instruction": "place the fan on the marked area",
    }


def test_formal_parity_rejects_unverifiable_external_vla_endpoint():
    args = argparse.Namespace(
        task_name="place_fan",
        lingbot_model_revision=CHECKPOINT_REVISION,
        robotwin_parity_debug=True,
        vla_endpoint="ws://localhost:8008",
    )

    with pytest.raises(ValueError, match="external --vla-endpoint"):
        _parse_config(args)


def test_cuda_device_split_is_backward_compatible():
    shared = argparse.Namespace(
        cuda_device=3,
        env_cuda_device=None,
        vla_cuda_device=None,
    )
    split = argparse.Namespace(
        cuda_device=None,
        env_cuda_device=0,
        vla_cuda_device=2,
    )

    assert _resolve_cuda_devices(shared) == ("3", "3")
    assert _resolve_cuda_devices(split) == ("0", "2")


def test_cuda_device_split_rejects_ambiguous_configuration():
    args = argparse.Namespace(
        cuda_device=3,
        env_cuda_device=0,
        vla_cuda_device=None,
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        _resolve_cuda_devices(args)


def test_hybrid_context_embeds_only_safe_semantic_references(tmp_path):
    workspace = tmp_path / "hybrid_workspace"
    recipe_dir = workspace / "recipe"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "place_fan_s0.json").write_text(
        json.dumps(
            {
                "task_language": "old clean-scene instruction",
                "semantic_recipe": {"phases": [{"goal": "place the fan"}]},
                "strategy_notes": {"roles": {"arm": "left"}},
                "evidence_status": "supported",
                "exact_seed": 100002,
                "saved_xyz": [0.1, 0.2, 0.3],
            }
        )
    )
    (recipe_dir / "recipe_place_fan_s0.jsonl").write_text(
        '{"action":"move_to","xyz":[0.1,0.2,0.3]}\n'
    )

    resolved, context = _resolve_hybrid_context(
        task_name="place_fan",
        workspace_value=str(workspace),
        robotwin_root=None,
    )

    assert resolved == str(workspace.resolve())
    assert "place the fan" in context
    assert '"arm": "left"' in context
    assert str(workspace) not in context
    assert "GUIDE" not in context
    assert "old clean-scene instruction" not in context
    assert "100002" not in context
    assert "saved_xyz" not in context
    assert "recipe_place_fan_s0.jsonl" not in context
    assert "[0.1,0.2,0.3]" not in context


@pytest.mark.parametrize(
    "legacy_value",
    [
        "write command.json",
        "connect with http://127.0.0.1:8000",
        "construct RoboTwinEnvClient",
        "read /mnt/public/legacy",
        "poll state_00.json",
    ],
)
def test_hybrid_context_rejects_legacy_recipe_content(tmp_path, legacy_value):
    workspace = tmp_path / "hybrid_workspace"
    recipe_dir = workspace / "recipe"
    recipe_dir.mkdir(parents=True)
    (recipe_dir / "place_fan_s0.json").write_text(
        json.dumps({"strategy_notes": {"unsafe": legacy_value}})
    )

    with pytest.raises(ValueError, match="forbidden legacy marker"):
        _resolve_hybrid_context(
            task_name="place_fan",
            workspace_value=str(workspace),
            robotwin_root=None,
        )


def test_curated_prompt_preserves_semantics_without_legacy_runtime_contract():
    prompt = "\n".join(
        str(value)
        for value in {
            **system_prompt(),
            **user_prompt(),
        }.values()
    )

    for semantic_rule in (
        "head view",
        "wrist view",
        "world-frame",
        "TCP",
        "complete current native task language",
        "handover",
        "remaining_steps",
        "eval_success",
        "sample_world_xyz",
        "query_world_map",
        "list_mcp_resources",
        "Do not call request_user_input",
        "update_plan",
        "process/session polling",
    ):
        assert semantic_rule in prompt
    for legacy_marker in (
        "Downloads GUIDE",
        "command.json",
        "done_NN.flag",
        "hybrid_driver.py",
        "RoboTwinEnvClient",
        "read GUIDE in full",
        "http://",
        "ws://",
        "{{output_dir}}",
        "{{instruction}}",
    ):
        assert legacy_marker not in prompt


def test_curated_prompt_requires_finish_for_every_planner_exit():
    success_contract = " ".join(str(system_prompt()["SUCCESS"]).split())

    assert "Every episode exit must be completed by calling" in success_contract
    assert "finish exactly once" in success_contract
    assert "including native success, native failure, budget" in success_contract
    assert "Never return a final answer" in success_contract
    assert "without that finish call" in success_contract


def test_curated_prompt_requires_tool_call_in_same_response():
    commitment = " ".join(
        str(system_prompt()["ACTION_COMMITMENT"]).split()
    )

    assert "emit that registered tool call in the same model response" in commitment
    assert "announcing, promising, or describing a call" in commitment
    assert "is not an action" in commitment
    assert "saved frame twice without a new mutation or render" in commitment


def test_untrusted_status_artifact_does_not_require_an_observation(tmp_path):
    state = dump_untrusted_status(
        output_dir=tmp_path,
        step_idx=4,
        status={"state_trustworthy": False, "termination_reason": "state_unknown"},
        log={"command": "move_to"},
    )

    assert state["frame_id"] is None
    assert state["artifacts"] == {}
    assert state["observation_unavailable"] == "state_unknown"
    assert (tmp_path / "state_04.json").exists()


def test_geometry_tool_schemas_do_not_add_back_project_alias():
    names = {spec["name"] for spec in TOOLS_SPEC}

    assert {"sample_world_xyz", "query_world_map"} <= names
    assert "back_project_batch" not in names


def test_robotwin_toolkit_does_not_expose_generic_file_discovery_tools():
    toolkit = RoboTwinToolkit.__new__(RoboTwinToolkit)
    Toolkit.__init__(toolkit)
    toolkit._remove_generic_file_tools()

    names = {spec["name"] for spec in toolkit.get_tools_spec()}
    assert "read_text_file" not in names
    assert "write_text_file" not in names
    assert "list_dir" not in names
    assert names == {"finish"}


def test_robotwin_runtime_tool_order_matches_preregistered_contract():
    toolkit = RoboTwinToolkit.__new__(RoboTwinToolkit)
    Toolkit.__init__(toolkit)
    toolkit._primitives = type(
        "Primitives",
        (),
        {"finish": lambda self, **kwargs: kwargs},
    )()
    toolkit._register_robotwin_tools()

    assert toolkit.get_tools_spec() == RoboTwinToolkit.contract_tool_specs()


def test_robotwin_native_success_allows_only_finish():
    toolkit = RoboTwinToolkit.__new__(RoboTwinToolkit)
    toolkit._latest_status = {
        "success": True,
        "state_trustworthy": True,
    }

    blocked = toolkit.before_execute_tool("lingbot_act", {"chunks": 1})

    assert blocked == {
        "error": "native_success_requires_finish",
        "success": True,
        "state_trustworthy": True,
        "required_next_tool": "finish",
        "message": (
            "Native success is already verified. No image, action, or "
            "other tool is allowed; call finish now."
        ),
    }
    assert toolkit.before_execute_tool(
        "finish",
        {"status": "success", "summary": "done"},
    ) is None


def test_robotwin_progress_signal_stays_internal():
    toolkit = RoboTwinToolkit.__new__(RoboTwinToolkit)
    item = {
        "tool": "lingbot_act",
        "result": {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "episode_status": {
                                "success": True,
                                "state_trustworthy": True,
                            }
                        }
                    ),
                }
            ]
        },
    }

    signal = toolkit.evaluate_progress(item)

    assert signal == {
        "environment_progress": True,
        "requires_finish": True,
        "progress_token": {"tool": "lingbot_act"},
    }
    assert "required_next_tool" not in signal


@pytest.mark.parametrize(
    "latest_status",
    [
        {"success": False, "state_trustworthy": True},
        {"success": True, "state_trustworthy": False},
        {
            "success": True,
            "state_trustworthy": False,
            "termination_reason": "state_unknown",
        },
    ],
)
def test_robotwin_untrusted_or_unsuccessful_status_does_not_lock_tools(
    latest_status,
):
    toolkit = RoboTwinToolkit.__new__(RoboTwinToolkit)
    toolkit._latest_status = latest_status

    assert toolkit.before_execute_tool("lingbot_act", {"chunks": 1}) is None


def test_codex_control_path_audit_distinguishes_harmless_shell(tmp_path):
    raw = tmp_path / "codex.stream.jsonl"
    events = [
        {
            "method": "item/completed",
            "payload": {"item": {"type": "commandExecution", "command": "pwd"}},
        },
        {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "commandExecution",
                    "command": "curl http://127.0.0.1:9000/env.execute_actions",
                }
            },
        },
        {
            "method": "item/completed",
            "payload": {
                "item": {
                    "type": "commandExecution",
                    "command": "python -c 'from robots.robotwin.env_client import RoboTwinEnvClient'",
                }
            },
        },
    ]
    raw.write_text("\n".join(json.dumps(event) for event in events))

    audit = audit_codex_control_path(raw)

    assert audit["control_path_violation"] == 2
    assert audit["manual_http_rpc_attempts"] == 1
    assert audit["direct_env_client_attempts"] == 1
    assert audit["shell_environment_mutation"] == 2


@pytest.mark.parametrize(
    "command",
    [
        "cat /tmp/hybrid_workspace/GUIDE.md",
        "python -c 'import json; print(json.load(open(\"state_00.json\")))'",
        "test -f done_03.flag",
    ],
)
def test_codex_control_path_audit_detects_legacy_file_discovery(
    tmp_path,
    command,
):
    raw = tmp_path / "codex.stream.jsonl"
    raw.write_text(
        json.dumps(
            {
                "payload": {
                    "item": {
                        "type": "commandExecution",
                        "command": command,
                    }
                }
            }
        )
    )

    audit = audit_codex_control_path(raw)

    assert audit["control_path_violation"] == 1
    assert audit["legacy_file_repl_attempts"] == 1
    assert audit["shell_environment_mutation"] == 1


def test_robotwin_finalize_run_marks_violation_as_hard_failure(tmp_path):
    raw = tmp_path / "codex.stream.jsonl"
    raw.write_text(
        json.dumps(
            {
                "payload": {
                    "item": {
                        "type": "commandExecution",
                        "command": "touch command.json",
                    }
                }
            }
        )
    )
    toolkit = RoboTwinToolkit.__new__(RoboTwinToolkit)
    toolkit._episode_id = "episode-test"
    toolkit._reset_precheck_s = 1.0
    toolkit._initial_capture_s = 2.0
    toolkit._telemetry_events = []
    toolkit._mutation_events = []
    toolkit._mutation_observed_count = 0
    toolkit._unknown_mutation_ids = set()
    toolkit._recipe = []
    toolkit._latest_status = {"take_action_cnt": 0, "step_lim": 400}
    toolkit._output_dir = tmp_path
    toolkit._primitives = type("Primitives", (), {"native_actions": 0})()

    telemetry = toolkit.finalize_run(
        planner_stats={"raw_stream_path": str(raw), "turns_used": 1},
        lifecycle={
            "env_server_startup_s": 3.0,
            "planner_wall_s": 4.0,
            "shutdown_s": 5.0,
        },
    )

    assert telemetry["control_path_violation"] == 1
    assert telemetry["failure_class"] == "control_path_failure"
    assert telemetry["mutation_summary"]["unknown_mutation_path"] == 0
    assert (tmp_path / "robotwin_telemetry.json").is_file()


def test_robotwin_finalize_run_marks_planner_no_action_loop_as_hard_failure(
    tmp_path,
):
    toolkit = _telemetry_only_toolkit(tmp_path)

    telemetry = toolkit.finalize_run(
        planner_stats={
            "max_consecutive_no_action_responses": 5,
            "planner_guard_warning_count": 2,
            "duplicate_image_guard_count": 1,
            "planner_no_action_loop": 1,
        },
        lifecycle={
            "planner_wall_s": 10.0,
            "planner_outcome": "planner_error",
        },
    )

    assert telemetry["failure_class"] == "planner_failure"
    assert telemetry["failure_reason"] == "planner_no_action_loop"
    assert telemetry["finish_origin"] == "guard_abort"


@pytest.mark.parametrize(
    ("planner_stats", "failure_reason"),
    [
        (
            {"planner_runtime_progress_adapter_error": 1},
            "planner_runtime_progress_adapter_error",
        ),
        (
            {
                "sdk_interrupt_did_not_converge": 1,
                "interrupt_count": 1,
                "interrupt_origin": "guard_abort",
                "interrupt_acknowledged": True,
                "stream_terminal_event_seen": False,
            },
            "sdk_interrupt_did_not_converge",
        ),
    ],
)
def test_robotwin_finalize_run_preserves_planner_runtime_failure(
    tmp_path,
    planner_stats,
    failure_reason,
):
    toolkit = _telemetry_only_toolkit(tmp_path)

    telemetry = toolkit.finalize_run(
        planner_stats=planner_stats,
        lifecycle={"planner_outcome": "planner_error"},
    )

    assert telemetry["failure_class"] == "planner_runtime_failure"
    assert telemetry["failure_reason"] == failure_reason
    assert telemetry["accepted_episode_success"] is False
    assert telemetry["interrupt_count"] == int(
        planner_stats.get("interrupt_count", 0)
    )


def test_robotwin_finalize_run_correlates_planner_tool_events(tmp_path):
    toolkit = _telemetry_only_toolkit(tmp_path)
    toolkit._telemetry_events = [
        {
            "episode_id": "episode-test",
            "turn_id": None,
            "tool_call_id": "tool-000001",
            "tool": "view_driver_state",
            "primitive_id": None,
            "mutation_id": None,
            "mutation_ids": [],
            "frame_id": None,
            "tool_dispatch_s": 0.0,
            "handler_s": 0.0,
            "primitive_native_s": 0.0,
            "status_query_s": 0.0,
            "observation_capture_s": 0.0,
            "artifact_write_s": 0.0,
            "other_s": 0.0,
            "error": False,
        }
    ]

    telemetry = toolkit.finalize_run(
        planner_stats={
            "mcp_tool_events": [
                {
                    "turn_id": "turn-000000",
                    "sdk_turn_id": "sdk-turn",
                    "sdk_tool_call_id": "sdk-call",
                    "tool": "view_driver_state",
                }
            ]
        },
        lifecycle={
            "planner_wall_s": 0.0,
            "planner_outcome": "planner_returned_without_finish",
        },
    )

    debug = [
        json.loads(line)
        for line in (
            tmp_path / "robotwin_telemetry.debug.jsonl"
        ).read_text().splitlines()
    ]
    run_record = debug[0]
    event = debug[1]
    assert event["turn_id"] == "turn-000000"
    assert event["sdk_turn_id"] == "sdk-turn"
    assert event["sdk_tool_call_id"] == "sdk-call"
    assert run_record["tool_event_correlation_errors"] == []
    assert run_record["termination"] == "planner_returned_without_finish"
    assert run_record["native_termination"] is None
    assert run_record["terminal_protocol_violation"] == 1
    assert run_record["hard_failure_reasons"] == [
        "planner_returned_without_finish"
    ]
    assert telemetry["finish_origin"] == "no_finish"
    assert run_record["hard_failure"] is True


def _telemetry_only_toolkit(tmp_path):
    toolkit = RoboTwinToolkit.__new__(RoboTwinToolkit)
    Toolkit.__init__(toolkit)
    toolkit._episode_id = "episode-test"
    toolkit._reset_precheck_s = 0.0
    toolkit._initial_capture_s = 0.0
    toolkit._telemetry_events = []
    toolkit._mutation_events = []
    toolkit._mutation_observed_count = 0
    toolkit._unknown_mutation_ids = set()
    toolkit._recipe = []
    toolkit._tool_details = {}
    toolkit._latest_status = {"take_action_cnt": 0, "step_lim": 400}
    toolkit._verified_finish_result = None
    toolkit._verified_finish_origin = None
    toolkit._planner_finish_count = 0
    toolkit._output_dir = tmp_path
    toolkit._primitives = type("Primitives", (), {"native_actions": 0})()
    toolkit._initializing = False
    toolkit._debug_telemetry = True
    return toolkit


def test_robotwin_uses_executed_native_finish_result(tmp_path):
    toolkit = _telemetry_only_toolkit(tmp_path)
    native_finish = {
        "_finish": True,
        "status": "failure",
        "requested_success": True,
        "success": False,
        "state_trustworthy": True,
        "native_status": {"success": False, "take_action_cnt": 10},
    }
    toolkit.add_tool(
        "finish",
        {"name": "finish", "input_schema": {"type": "object"}},
        lambda **_: native_finish,
    )

    dispatched = toolkit.execute_tool(
        "finish",
        {"status": "success", "summary": "looks done"},
    )
    resolved = _resolve_finish_result(
        "robotwin",
        toolkit,
        {"_finish": True, "status": "success", "summary": "looks done"},
    )

    assert dispatched.result == native_finish
    assert resolved == native_finish
    assert resolved is not native_finish
    assert toolkit._latest_status == native_finish["native_status"]
    assert toolkit._verified_finish_origin == "planner_finish"


def test_robotwin_native_success_with_real_finish_is_accepted(tmp_path):
    toolkit = _telemetry_only_toolkit(tmp_path)
    native_finish = {
        "_finish": True,
        "status": "success",
        "success": True,
        "state_trustworthy": True,
        "native_status": {
            "success": True,
            "state_trustworthy": True,
        },
    }
    toolkit.add_tool(
        "finish",
        {"name": "finish", "input_schema": {"type": "object"}},
        lambda **_: native_finish,
    )

    toolkit.execute_tool("finish", {"status": "success", "summary": "done"})
    telemetry = toolkit.finalize_run(
        planner_stats={"terminal_latched": True},
        lifecycle={"planner_outcome": "planner_finish"},
    )

    assert telemetry["native_success"] is True
    assert telemetry["finish_called"] is True
    assert telemetry["finish_origin"] == "planner_finish"
    assert telemetry["accepted_episode_success"] is True
    assert telemetry["failure_class"] is None


def test_robotwin_quit_does_not_count_as_accepted_finish(tmp_path):
    toolkit = _telemetry_only_toolkit(tmp_path)
    native_finish = {
        "_finish": True,
        "status": "failure",
        "success": False,
        "state_trustworthy": True,
        "native_status": {
            "success": True,
            "state_trustworthy": True,
        },
    }
    toolkit.add_tool(
        "quit",
        {"name": "quit", "input_schema": {"type": "object"}},
        lambda **_: native_finish,
    )

    toolkit.execute_tool("quit", {"status": "failure", "summary": "stop"})
    telemetry = toolkit.finalize_run(
        planner_stats={},
        lifecycle={"planner_outcome": "planner_returned_without_finish"},
    )

    assert toolkit.verified_finish_result is None
    assert telemetry["native_success"] is True
    assert telemetry["finish_called"] is False
    assert telemetry["finish_origin"] == "planner_quit"
    assert telemetry["accepted_episode_success"] is False
    assert telemetry["failure_class"] == "planner_failure"


def test_robotwin_duplicate_finish_is_planner_failure(tmp_path):
    toolkit = _telemetry_only_toolkit(tmp_path)
    native_finish = {
        "_finish": True,
        "status": "success",
        "success": True,
        "state_trustworthy": True,
        "native_status": {
            "success": True,
            "state_trustworthy": True,
        },
    }
    toolkit.add_tool(
        "finish",
        {"name": "finish", "input_schema": {"type": "object"}},
        lambda **_: native_finish,
    )

    for _ in range(2):
        toolkit.execute_tool("finish", {"status": "success", "summary": "done"})
    telemetry = toolkit.finalize_run(
        planner_stats={"terminal_latched": True},
        lifecycle={"planner_outcome": "planner_finish"},
    )

    assert telemetry["native_success"] is True
    assert telemetry["accepted_episode_success"] is False
    assert telemetry["failure_class"] == "planner_failure"
    assert telemetry["failure_reason"] == "duplicate_finish"


def test_robotwin_post_finish_guard_continued_is_distinct_planner_failure(
    tmp_path,
):
    toolkit = _telemetry_only_toolkit(tmp_path)
    native_finish = {
        "_finish": True,
        "status": "success",
        "success": True,
        "state_trustworthy": True,
        "native_status": {
            "success": True,
            "state_trustworthy": True,
        },
    }
    toolkit.add_tool(
        "finish",
        {"name": "finish", "input_schema": {"type": "object"}},
        lambda **_: native_finish,
    )

    toolkit.execute_tool("finish", {"status": "success", "summary": "done"})
    telemetry = toolkit.finalize_run(
        planner_stats={
            "planner_no_action_loop": 1,
            "terminal_latched": False,
        },
        lifecycle={"planner_outcome": "planner_error"},
    )

    assert telemetry["native_success"] is True
    assert telemetry["finish_origin"] == "planner_finish"
    assert telemetry["terminal_latched"] is False
    assert telemetry["accepted_episode_success"] is False
    assert telemetry["failure_class"] == "planner_failure"
    assert telemetry["failure_reason"] == "post_finish_guard_continued"


def test_robotwin_terminal_tool_failure_is_explicit_planner_failure(tmp_path):
    toolkit = _telemetry_only_toolkit(tmp_path)

    telemetry = toolkit.finalize_run(
        planner_stats={
            "terminal_tool_failure": 1,
            "terminal_latched": False,
        },
        lifecycle={"planner_outcome": "planner_error"},
    )

    assert telemetry["accepted_episode_success"] is False
    assert telemetry["terminal_latched"] is False
    assert telemetry["failure_class"] == "planner_failure"
    assert telemetry["failure_reason"] == "planner_terminal_tool_failed"


def test_robotwin_rejects_finish_signal_when_tool_did_not_complete(tmp_path):
    toolkit = _telemetry_only_toolkit(tmp_path)

    assert (
        _resolve_finish_result(
            "robotwin",
            toolkit,
            {"_finish": True, "status": "success", "summary": "looks done"},
        )
        is None
    )
    planner_finish = {"_finish": True, "status": "success"}
    assert _resolve_finish_result("libero", toolkit, planner_finish) is planner_finish


def test_mutation_observer_associates_toolkit_mutation_with_active_call(tmp_path):
    toolkit = _telemetry_only_toolkit(tmp_path)
    toolkit._active_tool_call_id = "tool-000001"

    toolkit._observe_mutation(
        {
            "method": "execute_actions",
            "mutation_id": "server:primitive:chunk:1",
            "phase": "submitted",
        }
    )

    event = toolkit._mutation_events[0]
    assert event["tool_call_id"] == "tool-000001"
    assert event["initialization"] is False


def test_initial_reset_mutation_is_not_an_unknown_control_path(tmp_path):
    toolkit = _telemetry_only_toolkit(tmp_path)
    toolkit._initializing = True
    toolkit._observe_mutation(
        {
            "method": "reset_episode",
            "mutation_id": "server:initial-reset",
            "phase": "submitted",
        }
    )

    telemetry = toolkit.finalize_run(
        planner_stats={},
        lifecycle={"planner_wall_s": 0.0},
    )

    assert telemetry["mutation_summary"]["unknown_mutation_path"] == 0
    assert telemetry["control_path_violation"] == 0
    assert telemetry["failure_class"] == "task_failure"


def test_runtime_mutation_without_toolkit_call_is_a_hard_failure(tmp_path):
    toolkit = _telemetry_only_toolkit(tmp_path)
    toolkit._observe_mutation(
        {
            "method": "execute_actions",
            "mutation_id": "server:bypass",
            "phase": "submitted",
        }
    )

    telemetry = toolkit.finalize_run(
        planner_stats={},
        lifecycle={"planner_wall_s": 0.0},
    )

    assert telemetry["mutation_summary"]["unknown_mutation_path"] == 1
    assert telemetry["control_path_violation"] == 1
    assert telemetry["failure_class"] == "control_path_failure"
