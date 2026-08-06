# Copyright 2026 The RPent Authors.

"""Codex SDK planner.

Mirror of ``claude_code.py``: a thin, SDK-first backend. ``solve()`` prepares
artifacts, drives one Codex SDK turn, and assembles a ``PlannerResult``.
RPent tools are exposed via the stdio MCP bridge configured through
``_codex_mcp_config_overrides``; this backend does not register tools in
process. Event rendering and stats live in a single ``_Recorder``.
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai_codex

from rpent.cli.tui import next_user_line
from rpent.planner.base import PlannerResult, strip_mcp_prefix
from rpent.planner.codex_runtime import (
    inspect_runtime,
    sha256_json,
    sha256_text,
    validate_runtime,
)
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("codex")

PROVIDER_ID = "rpent_proxy"
PROVIDER_ENV_KEY = "RPENT_CODEX_PROVIDER_KEY"
RUNTIME_MANIFEST_NAME = "codex_runtime_manifest.json"
ACTION_GUARD_WARNING = (
    "Recent model responses did not execute a registered environment tool. "
    "Do not announce or describe a future call. Call exactly one registered "
    "environment tool now, or use the environment's terminal tool if no "
    "meaningful action remains."
)
NATIVE_SUCCESS_FINISH_WARNING = (
    "The environment reports a trustworthy terminal success. Your next model "
    "response must contain exactly one real call to the required terminal tool. "
    "Do not inspect another observation, execute another action, or reply with "
    "text only."
)
GUARD_ACTIVE = "active"
GUARD_FINISH_PENDING = "finish_pending"
GUARD_TERMINATED = "terminated"

# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class CodexPlanner:
    """Planner backed by the OpenAI Codex Python SDK."""

    def __init__(
        self,
        *,
        output_dir: str,
        repo_root: str | Path | None = None,
        timeout_s: int = 600,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        enforce_action_guard: bool = False,
        action_guard_warn_after: int = 3,
        action_guard_abort_after: int = 5,
        dashboard: Any = None,
    ):
        """Initialize the Codex SDK backend."""
        self._output_dir = str(output_dir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._timeout_s = timeout_s
        self._extra_dirs = extra_dirs or []
        self._output_path = Path(output_path) if output_path else None
        self._model = model or os.environ.get("CODEX_MODEL", None)
        self._reasoning_effort = reasoning_effort or os.environ.get(
            "CODEX_REASONING_EFFORT",
            None,
        )
        self._enforce_action_guard = enforce_action_guard
        if action_guard_warn_after < 1:
            raise ValueError("action_guard_warn_after must be at least 1")
        if action_guard_abort_after <= action_guard_warn_after:
            raise ValueError(
                "action_guard_abort_after must be greater than "
                "action_guard_warn_after"
            )
        self._action_guard_warn_after = action_guard_warn_after
        self._action_guard_abort_after = action_guard_abort_after
        self._base_url = os.environ.get("CODEX_BASE_URL", None)
        self._api_key = os.environ.get("CODEX_API_KEY", None)
        self._dashboard = dashboard
        self._runtime_manifest_path = (
            Path(self._output_dir) / RUNTIME_MANIFEST_NAME
        )
        self._runtime_manifest = inspect_runtime(
            model=self._model,
            reasoning_effort=self._reasoning_effort,
            base_url=self._base_url,
            configured_binary=os.environ.get("CODEX_BIN"),
        )
        self._runtime_manifest["planner_action_guard"] = {
            "enabled": self._enforce_action_guard,
            "warn_after": self._action_guard_warn_after,
            "abort_after": self._action_guard_abort_after,
        }
        validate_runtime(self._runtime_manifest)
        self._write_runtime_manifest()

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue=None,
    ) -> PlannerResult:
        """Run one or more Codex SDK turns for the given prompt."""
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        self._runtime_manifest.update(
            {
                "prompt_sha256": sha256_text(prompt),
                "tool_schema_sha256": sha256_json(toolkit.get_tools_spec()),
            }
        )
        self._write_runtime_manifest()
        if self._output_path is None:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".out", prefix="codex_sdk_task_", delete=False
            ) as f:
                output_path = Path(f.name)
        else:
            output_path = self._output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_stream_path = output_path.with_suffix(output_path.suffix + ".stream.jsonl")
        last_message_path = output_path.with_suffix(output_path.suffix + ".last")
        recorder = _Recorder(
            max_turns=max_turns,
            dashboard=self._dashboard,
            enforce_action_guard=self._enforce_action_guard,
            action_guard_warn_after=self._action_guard_warn_after,
            action_guard_abort_after=self._action_guard_abort_after,
            progress_evaluator=(
                toolkit.evaluate_progress
                if self._enforce_action_guard
                else None
            ),
        )
        state: dict[str, Any] = {}

        # Start the in-thread MCP HTTP server so Codex can reach the
        # shared toolkit without spawning a subprocess.
        mcp_server = HttpMcpServer(toolkit)
        mcp_url = mcp_server.start()
        logger.info("mcp http endpoint: %s", mcp_url)

        model_desc = self._model or "(configured default)"
        logger.info("prompt: %d chars", len(prompt))
        logger.info("output_dir: %s", self._output_dir)
        logger.info(
            "invoking Codex SDK model %s (reasoning_effort=%s, timeout=%ds)",
            model_desc,
            self._reasoning_effort or "(configured default)",
            self._timeout_s,
        )
        logger.info("Codex runtime manifest: %s", self._runtime_manifest_path)

        started = time.time()
        worker = threading.Thread(
            target=self._run_session,
            args=(
                prompt,
                output_path,
                raw_stream_path,
                last_message_path,
                recorder,
                state,
                mcp_url,
                input_queue,
            ),
            name="codex-sdk",
            daemon=True,
        )
        worker.start()
        try:
            worker.join(timeout=self._timeout_s)

            error: str | None = None
            if worker.is_alive():
                error = f"Codex SDK timed out after {self._timeout_s}s"
                _interrupt(state)
                rendered = f"\n[codex-planner] {error}\n"
                with open(output_path, "a") as out_f:
                    out_f.write(rendered)
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "timeout", "message": error})
                logger.info(rendered.rstrip())
                worker.join(timeout=15)
            elif "error" in state:
                exc = state["error"]
                error = f"{type(exc).__name__}: {exc}"
                rendered = f"\n[codex-planner] {error}\n"
                with open(output_path, "a") as out_f:
                    out_f.write(rendered)
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "error", "message": error})
                logger.info(rendered.rstrip())
        finally:
            mcp_server.stop()

        elapsed = time.time() - started
        text = state.get("text", "") or output_path.read_text(errors="replace")
        error = error or recorder.error

        logger.info("Codex SDK finished in %.1fs", elapsed)
        logger.info("output: %s", output_path)
        logger.info("raw stream: %s", raw_stream_path)

        return PlannerResult(
            finish_result=recorder.finish_result,
            messages=[{"role": "codex_sdk", "content": text}],
            stats={
                "backend": "codex_sdk",
                "elapsed_s": round(elapsed, 1),
                "output_chars": len(text),
                "output_path": str(output_path),
                "raw_stream_path": str(raw_stream_path),
                "last_message_path": str(last_message_path),
                "last_message_chars": len(recorder.final_response or ""),
                **recorder.stats(),
            },
            error=error,
        )

    # -- internal session --------------------------------------------------

    def _run_session(
        self,
        prompt: str,
        output_path: Path,
        raw_stream_path: Path,
        last_message_path: Path,
        recorder: "_Recorder",
        state: dict[str, Any],
        mcp_url: str,
        input_queue: "queue.Queue[str | None] | None" = None,
    ) -> None:
        try:
            approval = openai_codex.ApprovalMode.deny_all
            sandbox = openai_codex.Sandbox.full_access
            chunks: list[str] = []
            with openai_codex.Codex(config=self._build_config(mcp_url)) as codex:
                state["codex"] = codex
                thread = codex.thread_start(
                    approval_mode=approval,
                    cwd=self._repo_root,
                    model=self._model,
                    sandbox=sandbox,
                )
                state["thread"] = thread

                with (
                    open(output_path, "w") as out_f,
                    open(raw_stream_path, "w") as raw_f,
                ):
                    write_lock = threading.Lock()

                    turn = thread.turn(
                        prompt,
                        approval_mode=approval,
                        cwd=self._repo_root,
                        model=self._model,
                        sandbox=sandbox,
                    )
                    state["turn"] = turn

                    stop_steer: threading.Event | None = None
                    if input_queue is not None:
                        stop_steer = threading.Event()

                        def _steer() -> None:
                            while True:
                                nxt = next_user_line(input_queue)
                                if stop_steer.is_set():
                                    return
                                if nxt is None:
                                    try:
                                        turn.interrupt()
                                    except Exception:
                                        pass
                                    return
                                rendered = f"\n[user] {nxt}\n"
                                with write_lock:
                                    chunks.append(rendered)
                                    out_f.write(rendered)
                                    out_f.flush()
                                logger.info(rendered.strip())
                                try:
                                    turn.steer(nxt)
                                except Exception as e:
                                    rendered = f"\n[codex-planner] steer failed: {e}\n"
                                    with write_lock:
                                        chunks.append(rendered)
                                        out_f.write(rendered)
                                        out_f.flush()
                                    logger.info(rendered.strip())
                                    return

                        threading.Thread(
                            target=_steer,
                            name="codex-steer",
                            daemon=True,
                        ).start()

                    try:
                        for event in turn.stream():
                            _write_jsonl(raw_f, _message_to_json(event))
                            if rendered := recorder.observe(event):
                                with write_lock:
                                    chunks.append(rendered)
                                    out_f.write(rendered)
                                    out_f.flush()
                                logger.info(rendered.strip())
                            if warning := recorder.consume_guard_warning():
                                rendered = f"\n[codex-guard] {warning}\n"
                                with write_lock:
                                    chunks.append(rendered)
                                    out_f.write(rendered)
                                    out_f.flush()
                                logger.warning(rendered.strip())
                                try:
                                    turn.steer(warning)
                                except Exception as error:
                                    logger.warning(
                                        "failed to steer Codex action guard: %s",
                                        error,
                                    )
                            if recorder.guard_state == GUARD_TERMINATED:
                                break
                            if recorder.abort_requested:
                                try:
                                    turn.interrupt()
                                except Exception:
                                    pass
                                break
                        recorder._complete_pending_response()
                        if recorder.abort_requested:
                            try:
                                turn.interrupt()
                            except Exception:
                                pass
                    finally:
                        if stop_steer is not None:
                            stop_steer.set()

            state["text"] = "".join(chunks)
            if recorder.final_response is not None:
                last_message_path.write_text(recorder.final_response)
        except Exception as e:
            state["error"] = e

    # -- config builder ----------------------------------------------------

    def _build_config(self, mcp_url: str) -> Any:
        env = {**os.environ}
        if self._api_key:
            env[PROVIDER_ENV_KEY] = self._api_key
        kwargs: dict[str, Any] = {
            "config_overrides": tuple(
                _codex_mcp_config_overrides(
                    mcp_url=mcp_url,
                    base_url=self._base_url,
                    reasoning_effort=self._reasoning_effort,
                )
            ),
            "cwd": self._repo_root,
            "env": env,
            # Disable experimental API features (namespace tools,
            # web_search, image_generation).  This forces the binary to
            # convert namespace MCP tools to function tools internally
            # while preserving its own name→namespace mapping so that
            # ``function_call`` responses can be routed through MCP.
            "experimental_api": False,
        }
        if codex_bin := os.environ.get("CODEX_BIN"):
            kwargs["codex_bin"] = codex_bin
        return openai_codex.CodexConfig(**kwargs)

    def _write_runtime_manifest(self) -> None:
        self._runtime_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self._runtime_manifest_path.write_text(
            json.dumps(self._runtime_manifest, indent=2, sort_keys=True) + "\n"
        )


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Pure adapter: consume Codex SDK events, emit text + accumulate stats."""

    max_turns: int
    dashboard: Any = None
    enforce_action_guard: bool = False
    action_guard_warn_after: int = 3
    action_guard_abort_after: int = 5
    progress_evaluator: Callable[[Any], dict[str, Any] | None] | None = None
    turns: int = 0
    tool_calls: int = 0
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_output_tokens": 0,
        }
    )
    final_response: str | None = None
    finish_result: dict[str, Any] | None = None
    error: str | None = None
    mcp_tool_events: list[dict[str, Any]] = field(default_factory=list)
    no_action_responses: int = 0
    max_consecutive_no_action_responses: int = 0
    no_tool_call_responses: int = 0
    max_consecutive_no_tool_call_responses: int = 0
    planner_guard_warning_count: int = 0
    duplicate_image_guard_count: int = 0
    native_success_finish_required_count: int = 0
    native_success_without_finish: bool = False
    terminal_tool_failure: bool = False
    abort_requested: bool = False
    guard_state: str = GUARD_ACTIVE
    logical_response_count: int = 0
    _response_item_types: set[str] = field(default_factory=set)
    _response_had_new_rpent_result: bool = False
    _response_had_new_image: bool = False
    _guard_warning_pending: str | None = None
    _progress_tokens: set[str] = field(default_factory=set)
    _image_view_counts: dict[str, int] = field(default_factory=dict)
    _warned_image_paths: set[str] = field(default_factory=set)
    _native_success_finish_required: bool = False
    _native_success_detected_in_response: bool = False
    _response_called_finish: bool = False
    _response_boundary_pending: bool = False

    def stats(self) -> dict[str, Any]:
        return {
            "turns_used": self.turns,
            "tool_calls": self.tool_calls,
            "mcp_tool_events": self.mcp_tool_events,
            "max_consecutive_no_action_responses": (
                self.max_consecutive_no_action_responses
            ),
            "max_consecutive_no_environment_progress_responses": (
                self.max_consecutive_no_action_responses
            ),
            "max_consecutive_no_tool_call_responses": (
                self.max_consecutive_no_tool_call_responses
            ),
            "planner_guard_warning_count": self.planner_guard_warning_count,
            "duplicate_image_guard_count": self.duplicate_image_guard_count,
            "native_success_finish_required_count": (
                self.native_success_finish_required_count
            ),
            "native_success_without_finish": int(
                self.native_success_without_finish
            ),
            "terminal_tool_failure": int(self.terminal_tool_failure),
            "terminal_latched": self.guard_state == GUARD_TERMINATED,
            "guard_state": self.guard_state,
            "logical_response_count": self.logical_response_count,
            "planner_no_action_loop": int(self.abort_requested),
            "planner_action_guard": {
                "enabled": self.enforce_action_guard,
                "warn_after": self.action_guard_warn_after,
                "abort_after": self.action_guard_abort_after,
            },
            **self.usage,
        }

    def observe(self, event: Any) -> str:
        method = str(_get(event, "method", ""))
        payload = _get(event, "payload")

        if method in {"thread/started", "turn/started"}:
            return f"[codex-system] {method}\n"
        if method == "item/started":
            self._observe_item_started(_get(payload, "item"))
            return ""
        if method == "item/completed":
            item = _get(payload, "item")
            self._observe_response_item(item)
            return self._render_item(
                item,
                sdk_turn_id=_optional_str(_get(payload, "turn_id")),
            )
        if method == "thread/tokenUsage/updated":
            self._set_usage(_get(payload, "token_usage"))
            if self._response_item_types:
                self._response_boundary_pending = True
            return ""
        if method == "turn/completed":
            self._complete_pending_response()
            return self._render_turn_completed(_get(payload, "turn"))
        if "requestApproval" in method:
            return f"[codex-approval] {method}\n"
        if method in {"error", "fatal"}:
            return f"[codex-error] {_short_json(_jsonable(payload), limit=500)}\n"
        return ""

    def consume_guard_warning(self) -> str | None:
        """Return one pending environment action reminder."""
        if self.guard_state != GUARD_ACTIVE:
            self._guard_warning_pending = None
            return None
        warning = self._guard_warning_pending
        self._guard_warning_pending = None
        return warning

    def _observe_item_started(self, item: Any) -> None:
        item = _unwrap(item)
        item_type = str(_get(item, "type", ""))
        if self._response_boundary_pending and item_type not in {
            "mcpToolCall",
            "dynamicToolCall",
            "commandExecution",
            "fileChange",
            "imageView",
        }:
            self._complete_pending_response()
        if item_type not in {"mcpToolCall", "dynamicToolCall"}:
            return
        server = str(_get(item, "server", "rpent"))
        tool_name = strip_mcp_prefix(str(_get(item, "tool", ""))).lower()
        if (
            self.enforce_action_guard
            and self.guard_state == GUARD_ACTIVE
            and server == "rpent"
            and tool_name == "finish"
        ):
            self.guard_state = GUARD_FINISH_PENDING
            self._guard_warning_pending = None

    def _observe_response_item(self, item: Any) -> None:
        item = _unwrap(item)
        item_type = str(_get(item, "type", ""))
        if not item_type or item_type == "userMessage":
            return
        self._response_item_types.add(item_type)
        if item_type in {"mcpToolCall", "dynamicToolCall"}:
            server = str(_get(item, "server", "rpent"))
            if server == "rpent":
                tool_name = strip_mcp_prefix(
                    str(_get(item, "tool", item_type))
                ).lower()
                progress = (
                    self.progress_evaluator(item)
                    if self.progress_evaluator is not None
                    else None
                )
                progress_token = (
                    progress.get("progress_token")
                    if isinstance(progress, dict)
                    else None
                )
                if progress_token is not None:
                    signature = sha256_json(progress_token)
                    if signature not in self._progress_tokens:
                        self._progress_tokens.add(signature)
                        self._response_had_new_rpent_result = True
                if tool_name == "finish":
                    self._response_called_finish = True
                    self._native_success_finish_required = False
                    terminal_succeeded = bool(
                        progress and progress.get("terminal_succeeded") is True
                    ) or (
                        str(_get(item, "status", "")) == "completed"
                        and _contains_finish_marker(_get(item, "result"))
                    )
                    if terminal_succeeded:
                        self.guard_state = GUARD_TERMINATED
                        self._guard_warning_pending = None
                        self.abort_requested = False
                        self.error = None
                    elif self.guard_state == GUARD_FINISH_PENDING:
                        self.guard_state = GUARD_ACTIVE
                        self.terminal_tool_failure = True
                        self.abort_requested = True
                        self.error = (
                            "planner_terminal_tool_failed: finish did not return "
                            "a successful terminal result"
                        )
                if (
                    tool_name != "finish"
                    and self.guard_state == GUARD_ACTIVE
                    and progress
                    and progress.get("requires_finish") is True
                ):
                    if not self._native_success_finish_required:
                        self.native_success_finish_required_count += 1
                    self._native_success_finish_required = True
                    self._native_success_detected_in_response = True
                    self._queue_guard_warning(NATIVE_SUCCESS_FINISH_WARNING)
            if self._response_boundary_pending:
                self._complete_pending_response()
            return
        if not self.enforce_action_guard or item_type != "imageView":
            return
        path = _optional_str(_get(item, "path") or _get(item, "filePath"))
        if path is None:
            return
        count = self._image_view_counts.get(path, 0) + 1
        self._image_view_counts[path] = count
        if count == 1:
            self._response_had_new_image = True
        if count == 3 and path not in self._warned_image_paths:
            self._warned_image_paths.add(path)
            self.duplicate_image_guard_count += 1
            self._queue_guard_warning(
                "The same saved environment observation has already been viewed "
                "twice without a new state. Do not load it again. Call one "
                "registered environment tool now, or use the terminal tool."
            )

    def _complete_pending_response(self) -> None:
        if not self._response_boundary_pending:
            return
        self._response_boundary_pending = False
        self._complete_model_response()

    def _complete_model_response(self) -> None:
        if not self._response_item_types:
            return
        self.logical_response_count += 1
        if self.guard_state != GUARD_ACTIVE:
            self._reset_response_state()
            return
        if self.enforce_action_guard and self._native_success_finish_required:
            if self._response_called_finish:
                self._native_success_finish_required = False
            elif self._native_success_detected_in_response:
                # The success-producing tool result and its immediate steer are
                # part of this response. Enforce finish on the next response.
                self._native_success_detected_in_response = False
            else:
                self.native_success_without_finish = True
                self.abort_requested = True
                self.error = (
                    "planner_no_action_loop: native_success_without_finish: "
                    "the response after trustworthy native success did not "
                    "call finish"
                )
                self._reset_response_state()
                return
        response_has_tool_call = bool(
            self._response_item_types
            & {
                "mcpToolCall",
                "dynamicToolCall",
                "commandExecution",
                "fileChange",
                "imageView",
            }
        )
        if self.enforce_action_guard and not response_has_tool_call:
            self.no_tool_call_responses += 1
            self.max_consecutive_no_tool_call_responses = max(
                self.max_consecutive_no_tool_call_responses,
                self.no_tool_call_responses,
            )
        else:
            self.no_tool_call_responses = 0

        response_made_environment_progress = (
            self._response_had_new_rpent_result or self._response_had_new_image
        )
        if self.enforce_action_guard and not response_made_environment_progress:
            self.no_action_responses += 1
            self.max_consecutive_no_action_responses = max(
                self.max_consecutive_no_action_responses,
                self.no_action_responses,
            )
            if self.no_action_responses == self.action_guard_warn_after:
                self._queue_guard_warning(ACTION_GUARD_WARNING)
            if self.no_action_responses >= self.action_guard_abort_after:
                self.abort_requested = True
                self.error = (
                    "planner_no_action_loop: "
                    f"{self.action_guard_abort_after} consecutive model responses "
                    "completed without an RPent tool result or a newly viewed "
                    "environment image"
                )
        else:
            self.no_action_responses = 0
        self._reset_response_state()

    def _reset_response_state(self) -> None:
        self._response_item_types.clear()
        self._response_had_new_rpent_result = False
        self._response_had_new_image = False
        self._response_called_finish = False

    def _queue_guard_warning(self, warning: str) -> None:
        if self._guard_warning_pending is not None:
            return
        self._guard_warning_pending = warning
        self.planner_guard_warning_count += 1

    # -- per-item handlers -------------------------------------------------

    def _render_item(self, item: Any, *, sdk_turn_id: str | None = None) -> str:
        item = _unwrap(item)
        item_type = str(_get(item, "type", ""))

        if item_type == "userMessage":
            text = _extract_text(_get(item, "content"))
            return f"\n[codex][user] {text}\n" if text else ""

        if item_type in {"hookPrompt", "plan"}:
            return ""

        if item_type == "agentMessage":
            text = str(_get(item, "text", "")).strip()
            if not text:
                return ""
            self.final_response = text
            self.turns += 1
            if self.dashboard is not None:
                self.dashboard.on_event({"type": "text", "text": text})
            return (
                f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n"
                f"[codex] {text}\n"
            )

        if item_type == "reasoning":
            text = _extract_text(_get(item, "summary") or _get(item, "content"))
            if text and self.dashboard is not None:
                self.dashboard.on_event({"type": "thinking", "text": text})
            return f"[codex-reasoning] {text}\n" if text else ""

        if item_type in {
            "mcpToolCall",
            "dynamicToolCall",
            "commandExecution",
            "fileChange",
        }:
            self.tool_calls += 1
            if item_type in {"mcpToolCall", "dynamicToolCall"}:
                name = strip_mcp_prefix(str(_get(item, "tool", item_type)))
                # Only RPent server calls have matching Toolkit dispatch events.
                # Codex-owned MCP calls such as list_mcp_resources remain visible
                # in the transcript and total tool count, but must not shift
                # the environment's event correlation sequence.
                server = str(_get(item, "server", "rpent"))
                if server == "rpent":
                    self.mcp_tool_events.append(
                        {
                            "sequence": len(self.mcp_tool_events) + 1,
                            "turn_id": f"turn-{self.turns:06d}",
                            "sdk_turn_id": sdk_turn_id,
                            "sdk_tool_call_id": _optional_str(_get(item, "id")),
                            "tool": name,
                        }
                    )
                    self._maybe_capture_finish(name, item)
            elif item_type == "commandExecution":
                name = str(_get(item, "command", item_type))
            else:
                name = "fileChange"
            payload = _summarise_item(item)
            if self.dashboard is not None:
                data = _jsonable(item)
                args = data.get("arguments", {}) if isinstance(data, dict) else {}
                self.dashboard.on_event(
                    {"type": "tool_call", "tool": name, "args": args}
                )
                self.dashboard.on_event(
                    {"type": "tool_result", "tool": name, "result": payload}
                )
            return f"[tool<-] {name}: {json.dumps(payload, ensure_ascii=False)}\n"

        return ""

    def _render_turn_completed(self, turn: Any) -> str:
        status = str(_get(_get(turn, "status"), "value", _get(turn, "status", "")))
        duration_ms = _get(turn, "duration_ms")
        if error := _get(turn, "error"):
            self.error = str(_get(error, "message", str(error)))

        parts = ["[codex-result]", status]
        if duration_ms is not None:
            parts.append(f"duration={float(duration_ms) / 1000:.1f}s")
        usage_line = (
            f"\n[usage] in={self.usage['total_input_tokens']} "
            f"cached={self.usage['total_cached_input_tokens']} "
            f"out={self.usage['total_output_tokens']} "
            f"reasoning={self.usage['total_reasoning_output_tokens']} "
            f"tool_calls={self.tool_calls}"
        )
        return " ".join(p for p in parts if p) + usage_line + "\n"

    # -- helpers -----------------------------------------------------------

    def _set_usage(self, usage: Any) -> None:
        if usage is None:
            return
        usage = _get(usage, "total", usage)
        self.usage = {
            "total_input_tokens": _int_attr(usage, "input_tokens"),
            "total_cached_input_tokens": _int_attr(usage, "cached_input_tokens"),
            "total_output_tokens": _int_attr(usage, "output_tokens"),
            "total_reasoning_output_tokens": _int_attr(
                usage, "reasoning_output_tokens"
            ),
        }
        if self.dashboard is not None:
            self.dashboard.on_usage(
                inp=self.usage["total_input_tokens"],
                out=self.usage["total_output_tokens"],
                tool_calls=self.tool_calls,
            )

    def _maybe_capture_finish(self, name: str, item: Any) -> None:
        if self.finish_result is not None:
            return
        if name.lower() != "finish":
            return
        data = _jsonable(item)
        if (
            not isinstance(data, dict)
            or data.get("status") != "completed"
            or not _contains_finish_marker(data.get("result"))
        ):
            return
        args = data.get("arguments") if isinstance(data, dict) else None
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = None
        if isinstance(args, dict):
            self.finish_result = {"_finish": True, **args}


# ---------------------------------------------------------------------------
# Codex config overrides
# ---------------------------------------------------------------------------


def _codex_mcp_config_overrides(
    *,
    mcp_url: str,
    base_url: str | None,
    reasoning_effort: str | None = None,
) -> list[str]:
    config: list[tuple[str, Any]] = [
        ("mcp_servers.rpent.url", mcp_url),
    ]
    if base_url:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            normalized = normalized + "/v1"
        config.extend(
            [
                ("model_provider", PROVIDER_ID),
                (f"model_providers.{PROVIDER_ID}.name", PROVIDER_ID),
                (f"model_providers.{PROVIDER_ID}.base_url", normalized),
                (f"model_providers.{PROVIDER_ID}.wire_api", "responses"),
                (f"model_providers.{PROVIDER_ID}.env_key", PROVIDER_ENV_KEY),
            ]
        )
    if reasoning_effort:
        config.append(("model_reasoning_effort", reasoning_effort))
    return [f"{key}={json.dumps(value)}" for key, value in config]


def _interrupt(state: dict[str, Any]) -> None:
    if (turn := state.get("turn")) is not None:
        try:
            turn.interrupt()
        except Exception:
            pass
    if (codex := state.get("codex")) is not None:
        try:
            codex.close()
        except Exception:
            pass


def _write_jsonl(file_obj, value: dict[str, Any]) -> None:
    file_obj.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    file_obj.flush()


def _message_to_json(message: Any) -> dict[str, Any]:
    return {
        "observed_monotonic_s": time.monotonic(),
        "method": _get(message, "method", ""),
        "payload": _jsonable(_get(message, "payload")),
    }


def _jsonable(value: Any) -> Any:
    value = _unwrap(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        return {"type": "bytes", "size": len(value)}
    return value


def _unwrap(value: Any) -> Any:
    return getattr(value, "root", value)


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _kind(value: Any) -> str:
    value = _unwrap(value)
    if isinstance(value, dict):
        return str(value.get("type") or value.get("kind") or "")
    return value.__class__.__name__


def _summarise_item(item: Any) -> dict[str, Any]:
    data = _jsonable(item)
    if not isinstance(data, dict):
        return {"size": _payload_size(data)}

    summary: dict[str, Any] = {}
    for key in ("path", "file_path", "filename", "status", "state", "exit_code"):
        value = data.get(key)
        if value not in (None, ""):
            summary[key] = value
    if command := (data.get("command") or data.get("cmd")):
        command_text = str(command)
        if len(command_text) > 200:
            command_text = command_text[:200] + f"...(+{len(command_text) - 200})"
        summary["command"] = command_text
    for key in ("content", "text", "output", "stdout", "stderr", "result"):
        if key in data and data[key] not in (None, ""):
            summary[f"{key}_size"] = _payload_size(data[key])

    if not summary:
        summary["keys"] = sorted(
            key for key in data if key not in {"content", "text", "output"}
        )
    return summary


def _extract_text(value: Any) -> str:
    value = _unwrap(value)
    if isinstance(value, str):
        text = value.strip()
        if "data:image" in text or (
            "base64" in text and ("image" in text or "iVBOR" in text)
        ):
            return "<image omitted>"
        return text
    if isinstance(value, list):
        parts = [_extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return ""


def _contains_finish_marker(value: Any) -> bool:
    value = _jsonable(value)
    if isinstance(value, dict):
        if value.get("_finish") is True:
            return True
        return any(_contains_finish_marker(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_finish_marker(child) for child in value)
    if isinstance(value, str):
        try:
            return _contains_finish_marker(json.loads(value))
        except (json.JSONDecodeError, TypeError):
            return False
    return False


def _payload_size(value: Any) -> int:
    return len(str(value or ""))


def _short_json(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit})"


def _int_attr(value: Any, key: str) -> int:
    return int(_get(value, key, 0) or 0)


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    rendered = str(value)
    return rendered or None
