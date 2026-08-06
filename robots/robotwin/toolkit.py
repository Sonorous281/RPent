# Copyright 2026 The RPent Authors.

"""RPent toolkit for Downloads-compatible RoboTwin hybrid control."""

from __future__ import annotations

import json
import time
import uuid
from functools import partial
from pathlib import Path
from typing import Any

from robots.robotwin import tools
from robots.robotwin.primitives import RoboTwinPrimitives
from robots.robotwin.telemetry import audit_codex_control_path
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_output_dir
from rpent.utils.templates import substitute

ROBOTWIN_TOOL_ORDER = (
    "view_driver_state",
    "sample_world_xyz",
    "query_world_map",
    "render",
    "lingbot_act",
    "move_to",
    "rotate_wrist",
    "set_gripper",
    "release",
    "reset",
    "finish",
    "quit",
)


def _iter_json_objects(value: Any):
    """Yield dictionaries from nested MCP result/content payloads."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith(("{", "[")):
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError:
                return
        else:
            return
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _iter_json_objects(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _iter_json_objects(child)


class RoboTwinToolkit(Toolkit):
    """Common RPent tools plus the parity-preserving RoboTwin primitives."""

    _SPECS = {spec["name"]: spec for spec in tools.TOOLS_SPEC}

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        video_path: str | None = None,
        dashboard: Any = None,
    ):
        super().__init__(dashboard=dashboard)
        self._episode_id = f"robotwin-{uuid.uuid4().hex}"
        self._video_path = video_path
        self._step_idx = 0
        self._output_dir = Path(get_output_dir())
        self._recipe: list[dict[str, Any]] = []
        self._tool_details: dict[str, dict[str, Any]] = {}
        self._telemetry_events: list[dict[str, Any]] = []
        self._mutation_events: list[dict[str, Any]] = []
        self._mutation_observed_count = 0
        self._unknown_mutation_ids: set[str] = set()
        self._latest_status: dict[str, Any] = {}
        self._verified_finish_result: dict[str, Any] | None = None
        self._verified_finish_origin: str | None = None
        self._planner_finish_count = 0
        self._initializing = True
        self._primitives = RoboTwinPrimitives(**primitives_kwargs)
        self._debug_telemetry = bool(
            getattr(self._primitives, "capture_debug_on_close", False)
        )
        self._primitives.env.mutation_observer = self._observe_mutation
        reset_started = time.monotonic()
        reset_result = self._primitives.reset()
        self._reset_precheck_s = time.monotonic() - reset_started
        if not reset_result.get("success", False):
            raise RuntimeError(f"initial RoboTwin reset failed: {reset_result}")
        self._register_robotwin_tools()
        capture_started = time.monotonic()
        self._capture("reset", reset_result, elapsed_s=0.0)
        self._initial_capture_s = time.monotonic() - capture_started
        self._initializing = False

    def _register_robotwin_tools(self) -> None:
        self._remove_generic_file_tools()
        self._tools.pop("finish", None)
        self.add_tool(
            "view_driver_state",
            self._SPECS["view_driver_state"],
            lambda step=None: tools.view_state(self._output_dir, step),
        )
        self.add_tool(
            "sample_world_xyz",
            self._SPECS["sample_world_xyz"],
            lambda **kwargs: tools.sample_world_xyz(self._output_dir, **kwargs),
        )
        self.add_tool(
            "query_world_map",
            self._SPECS["query_world_map"],
            lambda **kwargs: tools.query_world_map(self._output_dir, **kwargs),
        )
        self.add_tool("render", self._SPECS["render"], partial(self._step, "render"))
        for name in (
            "lingbot_act",
            "move_to",
            "rotate_wrist",
            "set_gripper",
            "release",
            "reset",
        ):
            self.add_tool(name, self._SPECS[name], partial(self._step, name))
        self.add_tool("finish", self._SPECS["finish"], self._primitives.finish)
        self.add_tool("quit", self._SPECS["quit"], self._primitives.finish)

    def before_execute_tool(
        self,
        name: str,
        input_dict: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Make trustworthy native success a finish-only terminal state."""
        if (
            name != "finish"
            and self._latest_status.get("success") is True
            and self._latest_status.get("state_trustworthy") is True
        ):
            return {
                "error": "native_success_requires_finish",
                "success": True,
                "state_trustworthy": True,
                "required_next_tool": "finish",
                "message": (
                    "Native success is already verified. No image, action, or "
                    "other tool is allowed; call finish now."
                ),
            }
        return None

    def evaluate_progress(self, result: Any) -> dict[str, Any] | None:
        """Expose terminal progress to the planner runtime only.

        This adapter deliberately returns a small internal signal.  The
        signal is never appended to a tool result, prompt, or tool schema.
        """
        tool_name = str(result.get("tool", "")).lower() if isinstance(result, dict) else ""
        payload = result.get("result") if isinstance(result, dict) else result
        if tool_name == "finish":
            terminal_succeeded = any(
                candidate.get("_finish") is True
                for candidate in _iter_json_objects(payload)
            )
            return {
                "terminal_tool": True,
                "terminal_succeeded": terminal_succeeded,
                "progress_token": {"tool": "finish"},
            }
        progress_token: dict[str, Any] = {"tool": tool_name}
        for candidate in _iter_json_objects(payload):
            for key in (
                "episode_generation",
                "executed_actions",
                "frame_id",
                "mutation_seq",
                "step_idx",
                "take_action_cnt",
                "termination_reason",
            ):
                value = candidate.get(key)
                if value is not None and isinstance(
                    value, (str, int, float, bool)
                ):
                    progress_token[key] = value
            if candidate.get("success") is True and candidate.get(
                "state_trustworthy"
            ) is True:
                return {
                    "environment_progress": True,
                    "requires_finish": True,
                    "progress_token": progress_token,
                }
        return (
            {
                "environment_progress": True,
                "progress_token": progress_token,
            }
            if len(progress_token) > 1
            else None
        )

    @classmethod
    def contract_tool_specs(cls) -> list[dict[str, Any]]:
        """Return the exact ordered schema exposed to the RoboTwin Planner."""
        return substitute([cls._SPECS[name] for name in ROBOTWIN_TOOL_ORDER])

    def _remove_generic_file_tools(self) -> None:
        """Keep legacy workspace discovery outside the RoboTwin tool surface."""
        for name in ("read_text_file", "write_text_file", "list_dir"):
            self._tools.pop(name, None)

    def _capture(
        self, command: str, result: dict[str, Any], *, elapsed_s: float
    ) -> dict[str, Any]:
        detail = self._active_detail()
        status_started = time.monotonic()
        status = self._primitives.status()
        detail["status_query_s"] = detail.get("status_query_s", 0.0) + (
            time.monotonic() - status_started
        )
        self._latest_status = status
        log = {
            "command": command,
            "result": result,
            "elapsed_s": elapsed_s,
        }
        if not status.get("state_trustworthy", True):
            return tools.dump_untrusted_status(
                output_dir=self._output_dir,
                step_idx=self._step_idx,
                status=status,
                log=log,
            )
        observation_started = time.monotonic()
        observation = self._primitives.env.capture_agent_observation()
        detail["observation_capture_s"] = detail.get("observation_capture_s", 0.0) + (
            time.monotonic() - observation_started
        )
        artifact_started = time.monotonic()
        state = tools.dump_observation(
            observation,
            output_dir=self._output_dir,
            step_idx=self._step_idx,
            status=status,
            log=log,
        )
        detail["artifact_write_s"] = detail.get("artifact_write_s", 0.0) + (
            time.monotonic() - artifact_started
        )
        detail["frame_id"] = state.get("frame_id")
        return state

    def _step(self, name: str, **kwargs) -> dict[str, Any]:
        started = time.monotonic()
        detail = self._active_detail()
        self._recipe.append({"action": name, **kwargs})
        native_started = time.monotonic()
        if name == "render":
            result = {"success": True}
        elif name == "reset":
            result = self._primitives.controlled_reset()
        else:
            result = getattr(self._primitives, name)(**kwargs)
        detail["primitive_native_s"] = time.monotonic() - native_started
        detail["primitive_id"] = self._primitives.last_primitive_id
        self._step_idx += 1
        state = self._capture(
            name,
            result,
            elapsed_s=round(time.monotonic() - started, 3),
        )
        state["result"] = result
        return state

    def _active_detail(self) -> dict[str, Any]:
        call_id = self._active_tool_call_id
        if call_id is None:
            return {}
        return self._tool_details.setdefault(call_id, {})

    def _observe_mutation(self, event: dict[str, Any]) -> None:
        self._mutation_observed_count += 1
        tool_call_id = self._active_tool_call_id
        mutation_id = event.get("mutation_id")
        if (
            not self._initializing
            and tool_call_id is None
            and isinstance(mutation_id, str)
        ):
            self._unknown_mutation_ids.add(mutation_id)
        if self._debug_telemetry:
            self._mutation_events.append(
                {
                    "episode_id": self._episode_id,
                    "tool_call_id": tool_call_id,
                    "initialization": self._initializing,
                    "observed_monotonic_s": time.monotonic(),
                    **event,
                }
            )

    @staticmethod
    def _find_mutation_id(value: Any) -> str | None:
        if isinstance(value, dict):
            mutation_id = value.get("mutation_id")
            if isinstance(mutation_id, str):
                return mutation_id
            for child in value.values():
                found = RoboTwinToolkit._find_mutation_id(child)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = RoboTwinToolkit._find_mutation_id(child)
                if found is not None:
                    return found
        return None

    def on_tool_event(self, event: dict[str, Any]) -> None:
        detail = self._tool_details.pop(event["tool_call_id"], {})
        mutation_ids = list(
            dict.fromkeys(
                mutation["mutation_id"]
                for mutation in self._mutation_events
                if mutation["tool_call_id"] == event["tool_call_id"]
            )
        ) if self._debug_telemetry else []
        handler_s = float(event.get("handler_s", 0.0))
        accounted = sum(
            float(detail.get(field, 0.0))
            for field in (
                "primitive_native_s",
                "status_query_s",
                "observation_capture_s",
                "artifact_write_s",
            )
        )
        result = event.get("result")
        if (
            event["name"] in {"finish", "quit"}
            and isinstance(result, dict)
            and result.get("_finish") is True
        ):
            self._verified_finish_result = dict(result)
            self._verified_finish_origin = (
                "planner_finish" if event["name"] == "finish" else "planner_quit"
            )
            if event["name"] == "finish":
                self._planner_finish_count += 1
            native_status = result.get("native_status")
            if isinstance(native_status, dict):
                self._latest_status = dict(native_status)
        if self._debug_telemetry:
            self._telemetry_events.append({
                "episode_id": self._episode_id,
                "turn_id": None,
                "tool_call_id": event["tool_call_id"],
                "tool": event["name"],
                "primitive_id": detail.get("primitive_id"),
                "mutation_id": (
                    mutation_ids[0] if mutation_ids else self._find_mutation_id(result)
                ),
                "mutation_ids": mutation_ids,
                "frame_id": detail.get("frame_id"),
                "tool_dispatch_s": round(float(event.get("tool_dispatch_s", 0.0)), 6),
                "handler_s": round(handler_s, 6),
                "primitive_native_s": round(
                    float(detail.get("primitive_native_s", 0.0)), 6
                ),
                "status_query_s": round(float(detail.get("status_query_s", 0.0)), 6),
                "observation_capture_s": round(
                    float(detail.get("observation_capture_s", 0.0)), 6
                ),
                "artifact_write_s": round(
                    float(detail.get("artifact_write_s", 0.0)), 6
                ),
                "other_s": round(max(0.0, handler_s - accounted), 6),
                "error": isinstance(result, dict) and bool(result.get("error")),
            })

    @property
    def verified_finish_result(self) -> dict[str, Any] | None:
        """Return the actual native-verified finish result, if one completed."""
        if (
            self._verified_finish_result is None
            or self._verified_finish_origin != "planner_finish"
        ):
            return None
        return dict(self._verified_finish_result)

    def finalize_run(
        self,
        *,
        planner_stats: dict[str, Any],
        lifecycle: dict[str, Any],
    ) -> dict[str, Any]:
        audit = audit_codex_control_path(planner_stats.get("raw_stream_path"))
        planner_tool_events = planner_stats.get("mcp_tool_events", [])
        correlation_errors: list[dict[str, Any]] = []
        for index, event in enumerate(self._telemetry_events):
            planner_event = (
                planner_tool_events[index]
                if isinstance(planner_tool_events, list)
                and index < len(planner_tool_events)
                and isinstance(planner_tool_events[index], dict)
                else None
            )
            if planner_event is None or planner_event.get("tool") != event["tool"]:
                correlation_errors.append(
                    {
                        "sequence": index + 1,
                        "tool": event["tool"],
                        "planner_tool": (
                            planner_event.get("tool") if planner_event else None
                        ),
                    }
                )
                continue
            event["turn_id"] = planner_event.get("turn_id")
            event["sdk_turn_id"] = planner_event.get("sdk_turn_id")
            event["sdk_tool_call_id"] = planner_event.get("sdk_tool_call_id")

        unknown_mutations = self._unknown_mutation_ids
        audit["unknown_mutation_path"] = len(unknown_mutations)
        if unknown_mutations:
            audit["control_path_violation"] += len(unknown_mutations)
        tool_dispatch_s = sum(
            event["tool_dispatch_s"] for event in self._telemetry_events
        )
        handler_s = sum(event["handler_s"] for event in self._telemetry_events)
        tool_total_s = tool_dispatch_s + handler_s
        planner_wall_s = float(lifecycle.get("planner_wall_s", 0.0))
        timings = {
            "env_server_startup_s": round(
                float(lifecycle.get("env_server_startup_s", 0.0)), 6
            ),
            "reset_precheck_s": round(self._reset_precheck_s, 6),
            "initial_capture_s": round(self._initial_capture_s, 6),
            "planner_wall_s": round(planner_wall_s, 6),
            "planner_think_s": round(max(0.0, planner_wall_s - tool_total_s), 6),
            "tool_total_s": round(tool_total_s, 6),
            "tool_dispatch_s": round(tool_dispatch_s, 6),
            "primitive_native_s": round(
                sum(event["primitive_native_s"] for event in self._telemetry_events),
                6,
            ),
            "status_query_s": round(
                sum(event["status_query_s"] for event in self._telemetry_events),
                6,
            ),
            "observation_capture_s": round(
                sum(event["observation_capture_s"] for event in self._telemetry_events),
                6,
            ),
            "artifact_write_s": round(
                sum(event["artifact_write_s"] for event in self._telemetry_events),
                6,
            ),
            "shutdown_s": round(float(lifecycle.get("shutdown_s", 0.0)), 6),
        }
        native_termination = self._latest_status.get("termination_reason")
        planner_outcome = str(
            lifecycle.get("planner_outcome", "planner_outcome_unknown")
        )
        run_termination = (
            native_termination
            if native_termination not in (None, "running")
            else planner_outcome
        )
        terminal_protocol_violation = int(
            planner_outcome == "planner_returned_without_finish"
        )
        planner_no_action_loop = int(
            planner_stats.get("planner_no_action_loop", 0)
        )
        native_success_without_finish = int(
            planner_stats.get("native_success_without_finish", 0)
        )
        terminal_tool_failure = int(
            planner_stats.get("terminal_tool_failure", 0)
        )
        terminal_latched = bool(planner_stats.get("terminal_latched", False))
        hard_failure_reasons: list[str] = []
        if audit["control_path_violation"] > 0:
            hard_failure_reasons.append("control_path_violation")
        if terminal_protocol_violation > 0:
            hard_failure_reasons.append("planner_returned_without_finish")
        if planner_no_action_loop > 0:
            hard_failure_reasons.append("planner_no_action_loop")
        if native_success_without_finish > 0:
            hard_failure_reasons.append("native_success_without_finish")
        state_trustworthy = self._latest_status.get("state_trustworthy") is True
        native_success = (
            self._latest_status.get("success") is True and state_trustworthy
        )
        finish_origin = getattr(self, "_verified_finish_origin", None)
        if finish_origin is None:
            finish_origin = (
                "guard_abort" if planner_no_action_loop else "no_finish"
            )
        finish_called = finish_origin == "planner_finish"
        finish_count = getattr(self, "_planner_finish_count", 0)
        post_finish_guard_continued = bool(
            finish_called and (planner_no_action_loop or not terminal_latched)
        )
        if finish_count > 1:
            hard_failure_reasons.append("duplicate_finish")
        if terminal_tool_failure:
            hard_failure_reasons.append("planner_terminal_tool_failed")
        if post_finish_guard_continued:
            hard_failure_reasons.append("post_finish_guard_continued")
        if (
            terminal_tool_failure
            or post_finish_guard_continued
            or native_success_without_finish
            or terminal_protocol_violation
            or planner_no_action_loop
            or finish_count > 1
        ):
            failure_class = "planner_failure"
            failure_reason = (
                "planner_terminal_tool_failed"
                if terminal_tool_failure
                else (
                    "post_finish_guard_continued"
                    if post_finish_guard_continued
                    else (
                        "native_success_without_finish"
                        if native_success_without_finish
                        else (
                            "planner_returned_without_finish"
                            if terminal_protocol_violation
                            else (
                                "planner_no_action_loop"
                                if planner_no_action_loop
                                else "duplicate_finish"
                            )
                        )
                    )
                )
            )
        elif audit["control_path_violation"] > 0:
            failure_class = "control_path_failure"
            failure_reason = "control_path_violation"
        elif hard_failure_reasons:
            failure_class = "infra_failure"
            failure_reason = hard_failure_reasons[0]
        elif not native_success:
            failure_class = "task_failure"
            failure_reason = "native_eval_success_false"
        else:
            failure_class = None
            failure_reason = None
        summary = {
            "native_success": native_success,
            "accepted_episode_success": bool(
                native_success
                and finish_called
                and finish_count == 1
                and terminal_latched
                and state_trustworthy
                and not hard_failure_reasons
            ),
            "finish_called": finish_called,
            "finish_origin": finish_origin,
            "terminal_latched": terminal_latched,
            "state_trustworthy": state_trustworthy,
            "failure_class": failure_class,
            "failure_reason": failure_reason,
            "control_path_violation": int(audit["control_path_violation"]),
            "mutation_summary": {
                "observed": self._mutation_observed_count,
                "unknown_mutation_path": len(unknown_mutations),
            },
        }
        path = self._output_dir / "robotwin_telemetry.json"
        if getattr(self, "_debug_telemetry", False):
            debug_path = self._output_dir / "robotwin_telemetry.debug.jsonl"
            debug_records = [
                {
                    "type": "run",
                    **summary,
                    "episode_id": self._episode_id,
                    "timings": timings,
                    "terminal_protocol_violation": terminal_protocol_violation,
                    "hard_failure_reasons": hard_failure_reasons,
                    "hard_failure": bool(hard_failure_reasons),
                    "planner": {
                        "turns": int(planner_stats.get("turns_used", 0)),
                        "planner_no_action_loop": planner_no_action_loop,
                        "native_success_without_finish": (
                            native_success_without_finish
                        ),
                    },
                    "audit": audit,
                    "termination": run_termination,
                    "native_termination": native_termination,
                    "planner_outcome": planner_outcome,
                    "tool_event_correlation_errors": correlation_errors,
                },
                *(
                    {"type": "tool", **event}
                    for event in self._telemetry_events
                ),
                *(
                    {"type": "mutation", **event}
                    for event in self._mutation_events
                ),
            ]
            debug_path.write_text(
                "".join(
                    json.dumps(record, default=tools._json_default) + "\n"
                    for record in debug_records
                )
            )
            summary["debug_path"] = str(debug_path)
        path.write_text(json.dumps(summary, indent=2, default=tools._json_default))
        return summary

    def write_recipe(self, recipe_tag: str) -> str:
        path = self._output_dir / f"recipe_{recipe_tag}.jsonl"
        lines = [json.dumps(command, separators=(",", ":")) for command in self._recipe]
        path.write_text("\n".join(lines) + ("\n" if lines else ""))
        return str(path)

    def close(self) -> None:
        final_status_path = self._output_dir / "robotwin_final_status.json"
        try:
            final_status = self._primitives.status()
            self._latest_status = dict(final_status)
            final_status_path.write_text(
                json.dumps(
                    {
                        "capture_failed": False,
                        "fresh": True,
                        "status": final_status,
                    },
                    indent=2,
                    default=tools._json_default,
                )
            )
        except Exception as error:  # noqa: BLE001
            final_status_path.write_text(
                json.dumps(
                    {
                        "capture_failed": True,
                        "fresh": False,
                        "error": f"{type(error).__name__}: {error}",
                    },
                    indent=2,
                )
            )
        if not self._primitives.capture_debug_on_close:
            return
        path = self._output_dir / "robotwin_debug_state.json"
        try:
            debug_state = self._primitives.env.capture_debug_state()
            path.write_text(
                json.dumps(debug_state, indent=2, default=tools._json_default)
            )
        except Exception as error:  # noqa: BLE001
            path.write_text(
                json.dumps(
                    {
                        "capture_failed": True,
                        "error": f"{type(error).__name__}: {error}",
                    },
                    indent=2,
                )
            )
