# Copyright 2026 The RPent Authors.

"""RoboTwin-only Planner control-path audit helpers."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _walk(value: Any):
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def _command_executions(raw_stream_path: str | Path | None) -> list[str]:
    if not raw_stream_path:
        return []
    path = Path(raw_stream_path)
    if not path.is_file():
        return []
    commands: list[str] = []
    for line in path.read_text(errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for item in _walk(event):
            if item.get("type") != "commandExecution":
                continue
            command = item.get("command") or item.get("cmd")
            if command is not None:
                commands.append(str(command))
    return commands


def _classify_command(command: str) -> list[str]:
    normalized = command.lower()
    categories: list[str] = []
    if "command.json" in normalized:
        categories.append("command_json")
    if any(
        marker in normalized
        for marker in (
            "final.json",
            "hybrid_driver.py",
            "file-based repl",
            "hybrid_workspace",
            "guide.md",
        )
    ) or re.search(r"\b(?:state|log)_\d+\.json\b|\bdone_\d+\.flag\b", normalized):
        categories.append("legacy_file_repl")
    if any(
        marker in normalized
        for marker in (
            "robotwinenvclient",
            "envclient(",
            "env_client.py",
            "_rpc_client(",
        )
    ):
        categories.append("direct_env_client")
    network_program = re.search(
        r"(^|[;&|]\s*|\s)(curl|wget|nc|ncat|socat)\s", normalized
    )
    python_network = ("requests." in normalized or "httpx." in normalized) and (
        "http://" in normalized or "https://" in normalized
    )
    if network_program or python_network:
        categories.append("manual_http_rpc")
    mutation_terms = (
        "execute_actions",
        "execute_qpos_updates",
        "reset_episode",
        "set_gripper",
        "lingbot_act",
        "move_to",
    )
    launches_code = re.search(r"(^|\s)(python[0-9.]*|uv\s+run)\s", normalized)
    if categories or (
        launches_code and any(term in normalized for term in mutation_terms)
    ):
        categories.append("shell_environment_mutation")
    return list(dict.fromkeys(categories))


def audit_codex_control_path(
    raw_stream_path: str | Path | None,
) -> dict[str, Any]:
    """Classify Codex shell commands without inspecting command output."""
    commands = _command_executions(raw_stream_path)
    violations: list[dict[str, Any]] = []
    counts = {
        "shell_environment_mutation": 0,
        "manual_http_rpc_attempts": 0,
        "direct_env_client_attempts": 0,
        "command_json_attempts": 0,
        "legacy_file_repl_attempts": 0,
        "unknown_mutation_path": 0,
    }
    count_keys = {
        "shell_environment_mutation": "shell_environment_mutation",
        "manual_http_rpc": "manual_http_rpc_attempts",
        "direct_env_client": "direct_env_client_attempts",
        "command_json": "command_json_attempts",
        "legacy_file_repl": "legacy_file_repl_attempts",
    }
    for command in commands:
        categories = _classify_command(command)
        if not categories:
            continue
        violations.append({"command": command, "categories": categories})
        for category in categories:
            counts[count_keys[category]] += 1
    return {
        **counts,
        "control_path_violation": len(violations),
        "violations": violations,
    }
