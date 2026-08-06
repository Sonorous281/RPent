# Copyright 2026 The RPent Authors.

"""Codex runtime identity inspection and preregistration checks."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

PROVIDER_ID = "rpent_proxy"
RUNTIME_EXPECTATIONS = {
    "model": "RPENT_CODEX_EXPECTED_MODEL",
    "reasoning_effort": "RPENT_CODEX_EXPECTED_REASONING_EFFORT",
    "sdk_version": "RPENT_CODEX_EXPECTED_SDK_VERSION",
    "binary_path": "RPENT_CODEX_EXPECTED_BIN_PATH",
    "binary_version": "RPENT_CODEX_EXPECTED_BIN_VERSION",
    "provider": "RPENT_CODEX_EXPECTED_PROVIDER",
}


def inspect_runtime(
    *,
    model: str | None,
    reasoning_effort: str | None,
    base_url: str | None,
    configured_binary: str | None,
) -> dict[str, Any]:
    """Inspect the exact Codex SDK and binary selected for this run."""
    if configured_binary:
        binary = Path(configured_binary).expanduser()
    else:
        from codex_cli_bin import bundled_codex_path

        binary = bundled_codex_path()
    if not binary.is_file():
        raise FileNotFoundError(f"Codex binary not found: {binary}")

    configured_path = str(binary.absolute())
    resolved_path = str(binary.resolve())
    try:
        completed = subprocess.run(
            [configured_path, "--version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise RuntimeError(
            f"failed to inspect Codex binary {configured_path}: {error}"
        ) from error
    version_output = (completed.stdout or completed.stderr).strip()
    version_match = re.search(
        r"\b(\d+\.\d+\.\d+(?:[-+][^\s]+)?)\b",
        version_output,
    )
    if version_match is None:
        raise RuntimeError(
            f"unable to parse Codex binary version from: {version_output!r}"
        )

    try:
        sdk_version = importlib.metadata.version("openai-codex")
    except importlib.metadata.PackageNotFoundError:
        sdk_version = "unknown"

    return {
        "model": model,
        "reasoning_effort": reasoning_effort,
        "sdk_package": "openai-codex",
        "sdk_version": sdk_version,
        "binary_source": "configured" if configured_binary else "bundled",
        "binary_path": configured_path,
        "binary_resolved_path": resolved_path,
        "binary_version": version_match.group(1),
        "binary_version_output": version_output,
        "provider": PROVIDER_ID if base_url else "codex_default",
        "prompt_sha256": None,
        "tool_schema_sha256": None,
    }


def validate_runtime(runtime: dict[str, Any]) -> None:
    """Fail fast when the inspected runtime differs from preregistration."""
    mismatches: list[str] = []
    for runtime_field, env_name in RUNTIME_EXPECTATIONS.items():
        expected = os.environ.get(env_name)
        if expected is None:
            continue
        actual = runtime.get(runtime_field)
        if runtime_field == "binary_path":
            expected = str(Path(expected).expanduser().resolve())
            actual = runtime.get("binary_resolved_path")
        if str(actual) != expected:
            mismatches.append(
                f"{runtime_field}: expected {expected!r}, got {actual!r}"
            )
    if mismatches:
        raise RuntimeError("Codex runtime contract mismatch: " + "; ".join(mismatches))


def sha256_text(value: str) -> str:
    """Return the SHA256 of UTF-8 text."""
    return hashlib.sha256(value.encode()).hexdigest()


def sha256_json(value: Any) -> str:
    """Return the SHA256 of canonical JSON."""
    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_text(canonical)
