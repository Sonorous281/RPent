# Copyright 2026 The RPent Authors.

"""RoboTwin environment extension backed by RLinf RoboTwinEnv."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.robotwin.prompt_bundle import system_prompt, user_prompt
from rpent.envs.env_spec import EnvSpec, RunConfig
from rpent.envs.prompt_bundle import PromptBundle
from rpent.utils.config import get_repo_root

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon

CONTRACT_VERSION = "robotwin-agent-v1"
COMPATIBILITY_ID = "robotwin-rpent-downloads-2026-07-31-v1"
MODEL_MANIFEST_PATH = Path(__file__).with_name("model_manifest.json")
MODEL_MANIFEST = json.loads(MODEL_MANIFEST_PATH.read_text())
CHECKPOINT_REVISION = MODEL_MANIFEST["revision"]
TASK_CONFIGS = ("demo_clean", "demo_randomized")
SEED_MODES = ("exact", "walk")
SEMANTIC_RECIPE_FIELDS = (
    "semantic_recipe",
    "strategy_notes",
    "what_failed",
    "vla_recipe",
    "recovery_notes",
    "evidence_status",
    "eval_success_scope",
)
MAX_SEMANTIC_RECIPE_BYTES = 32_000
LEGACY_RECIPE_MARKERS = (
    "command.json",
    "done_",
    "final.json",
    "file-based repl",
    "hybrid_workspace",
    "envclient",
    "http://",
    "https://",
    "ws://",
    "curl ",
    "/users/",
    "/mnt/",
)
LEGACY_RECIPE_PATTERNS = (
    re.compile(r"\bstate_\d+\.json\b", re.IGNORECASE),
    re.compile(r"\b(?:https?|wss?)://", re.IGNORECASE),
    re.compile(r"\b(?:robo)?twinenvclient\s*\(", re.IGNORECASE),
)
SEMANTIC_RECIPE_TYPES = {
    "semantic_recipe": dict,
    "strategy_notes": dict,
    "what_failed": dict,
    "vla_recipe": (str, dict, list),
    "recovery_notes": (str, dict, list),
    "evidence_status": str,
    "eval_success_scope": str,
}
SCENE_COLOR_TERMS = (
    "black",
    "blue",
    "brown",
    "cyan",
    "gray",
    "green",
    "grey",
    "orange",
    "pink",
    "purple",
    "red",
    "white",
    "yellow",
)


def _redact_conflicting_scene_colors(
    value: Any,
    *,
    current_instruction: str | None,
) -> tuple[Any, bool]:
    """Remove stale scene colors while retaining recipe strategy structure."""
    instruction_colors = {
        color
        for color in SCENE_COLOR_TERMS
        if re.search(rf"\b{re.escape(color)}\b", current_instruction or "", re.I)
    }
    redacted = False

    def _walk(item: Any) -> Any:
        nonlocal redacted
        if isinstance(item, dict):
            return {key: _walk(child) for key, child in item.items()}
        if isinstance(item, list):
            return [_walk(child) for child in item]
        if not isinstance(item, str):
            return item
        result = item
        for color in SCENE_COLOR_TERMS:
            if color in instruction_colors:
                continue
            result, count = re.subn(
                rf"\b{re.escape(color)}\b",
                "<scene-color>",
                result,
                flags=re.IGNORECASE,
            )
            redacted = redacted or count > 0
        return result

    return _walk(value), redacted


def _initial_native_seed(seed: int, seed_mode: str) -> int:
    """Translate the Downloads seed CLI contract into a native episode seed."""
    if seed_mode == "exact":
        return int(seed)
    if seed_mode == "walk":
        return 100000 * (1 + int(seed))
    raise ValueError(f"unsupported seed mode: {seed_mode!r}")


def _resolve_hybrid_context(
    *,
    task_name: str,
    workspace_value: str | None,
    robotwin_root: str | None,
    instruction: str | None = None,
) -> tuple[str | None, str]:
    """Load only allowlisted semantic recipe data from the baseline workspace."""
    workspace = (
        Path(workspace_value).expanduser().resolve() if workspace_value else None
    )
    if workspace is None and robotwin_root:
        candidate = Path(robotwin_root).expanduser().resolve() / "hybrid_workspace"
        if candidate.is_dir():
            workspace = candidate
    if workspace is None:
        return (
            None,
            "Same-task semantic recipe: not available. Follow the embedded "
            "RoboTwin semantic rules and current observations.",
        )
    if not workspace.is_dir():
        raise ValueError(f"RoboTwin hybrid workspace not found: {workspace}")

    recipe = workspace / "recipe" / f"{task_name}_s0.json"
    if recipe.is_file():
        recipe_payload = json.loads(recipe.read_text())
        if not isinstance(recipe_payload, dict):
            raise ValueError(
                f"RoboTwin semantic recipe must be a JSON object: {recipe}"
            )
        semantic_recipe = {
            key: recipe_payload[key]
            for key in SEMANTIC_RECIPE_FIELDS
            if key in recipe_payload
        }
        semantic_recipe, scene_colors_redacted = _redact_conflicting_scene_colors(
            semantic_recipe,
            current_instruction=instruction,
        )
        for key, value in semantic_recipe.items():
            expected_type = SEMANTIC_RECIPE_TYPES[key]
            if not isinstance(value, expected_type):
                expected_names = (
                    ", ".join(item.__name__ for item in expected_type)
                    if isinstance(expected_type, tuple)
                    else expected_type.__name__
                )
                raise ValueError(
                    f"RoboTwin semantic recipe field {key!r} must be "
                    f"{expected_names}, got {type(value).__name__}"
                )
        recipe_text = json.dumps(semantic_recipe, ensure_ascii=False)
        if len(recipe_text.encode("utf-8")) > MAX_SEMANTIC_RECIPE_BYTES:
            raise ValueError(
                "RoboTwin semantic recipe exceeds the 32000-byte context limit"
            )
        normalized = recipe_text.lower()
        marker = next(
            (item for item in LEGACY_RECIPE_MARKERS if item in normalized),
            None,
        )
        pattern = next(
            (
                item.pattern
                for item in LEGACY_RECIPE_PATTERNS
                if item.search(recipe_text)
            ),
            None,
        )
        forbidden = marker or pattern
        if forbidden is not None:
            raise ValueError(
                "RoboTwin semantic recipe contains a forbidden legacy marker: "
                f"{forbidden!r}"
            )
        compact_recipe = json.dumps(
            semantic_recipe,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        recipe_sha256 = hashlib.sha256(compact_recipe.encode("utf-8")).hexdigest()
        recipe_text = json.dumps(
            semantic_recipe,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
    else:
        recipe_text = "not available"
        recipe_sha256 = None
        scene_colors_redacted = False
    context = "\n".join(
        [
            "Same-task semantic recipe (allowlisted fields only):",
            recipe_text,
            f"Injected semantic recipe SHA256: {recipe_sha256 or 'not available'}",
            (
                "Conflicting or unverifiable historical scene colors were "
                "redacted: "
                f"{str(scene_colors_redacted).lower()}"
            ),
            "",
            "Recipe and memory are clean-scene semantic priors only. Historical "
            "coordinates, pixels, object poses, and action traces are not exposed. "
            "Re-localize everything from the current episode.",
        ]
    )
    return str(workspace), context


def get_env_spec() -> EnvSpec:
    return EnvSpec(
        name="robotwin",
        prompts=PromptBundle(system=system_prompt, user=user_prompt),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_runtime=_init_runtime,
        planner_progress_guard=True,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    video_path: str | None = None,
    dashboard: Any = None,
):
    from robots.robotwin.toolkit import RoboTwinToolkit

    return RoboTwinToolkit(
        primitives_kwargs=primitives_kwargs,
        video_path=video_path,
        dashboard=dashboard,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    required = not use_dashboard
    parser.add_argument("--task-name", required=required)
    parser.add_argument("--seed", type=int, default=100002)
    parser.add_argument(
        "--task-config",
        choices=TASK_CONFIGS,
        default="demo_randomized",
        help="Native RoboTwin task YAML. Formal effect parity uses demo_randomized.",
    )
    parser.add_argument(
        "--seed-mode",
        choices=SEED_MODES,
        default="exact",
        help=(
            "exact uses --seed as the native seed; walk matches Downloads and "
            "starts from 100000 * (1 + --seed)."
        ),
    )
    parser.add_argument(
        "--allow-infeasible",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For an exact seed, continue after a failed expert feasibility "
            "precheck. Formal Downloads evaluation enables this."
        ),
    )
    parser.add_argument("--instruction-type", default="seen")
    parser.add_argument(
        "--instruction",
        default=None,
        help="Fixed native task instruction from the formal seed table.",
    )
    parser.add_argument(
        "--robotwin-hybrid-workspace",
        default=os.environ.get("ROBOTWIN_HYBRID_WORKSPACE"),
        help=(
            "Downloads hybrid_workspace containing GUIDE.md, recipe/, and memory/. "
            "Defaults to <robotwin-root>/hybrid_workspace when present."
        ),
    )
    parser.add_argument(
        "--robotwin-root",
        default=os.environ.get("ROBOTWIN_ROOT"),
    )
    parser.add_argument(
        "--robotwin-assets-root",
        default=os.environ.get("ROBOTWIN_ASSETS_ROOT"),
        help=(
            "Root containing assets/ for the pinned RoboTwin snapshot. "
            "Defaults to --robotwin-root for a complete checkout."
        ),
    )
    parser.add_argument(
        "--curobo-root",
        default=os.environ.get("ROBOTWIN_CUROBO_ROOT"),
        help=(
            "Clean NVlabs/curobo checkout at the Downloads-pinned revision. "
            "Defaults to <robotwin-root>/envs/curobo."
        ),
    )
    parser.add_argument("--env-endpoint", default=None)
    parser.add_argument("--vla-endpoint", default=None)
    parser.add_argument("--vla-root", default=os.environ.get("VLA_ROOT"))
    parser.add_argument(
        "--env-python",
        default=os.environ.get("ROBOTWIN_ENV_PYTHON", sys.executable),
    )
    parser.add_argument(
        "--vla-python",
        default=os.environ.get("LINGBOT_PY", sys.executable),
    )
    parser.add_argument(
        "--lingbot-model-path", default=os.environ.get("LINGBOT_MODEL_PATH")
    )
    parser.add_argument("--lingbot-model-revision", default=CHECKPOINT_REVISION)
    parser.add_argument("--cuda-device", default=None)
    parser.add_argument(
        "--env-cuda-device",
        default=None,
        help="Physical CUDA device used by the RoboTwin EnvServer.",
    )
    parser.add_argument(
        "--vla-cuda-device",
        default=None,
        help="Physical CUDA device used by the LingBot VLA server.",
    )
    parser.add_argument("--allow-reset", action="store_true")
    parser.add_argument(
        "--planner-guard-warn-after",
        type=int,
        default=3,
        help=(
            "Steer the RoboTwin Codex planner after this many consecutive "
            "responses without environment progress."
        ),
    )
    parser.add_argument(
        "--planner-guard-abort-after",
        type=int,
        default=5,
        help=(
            "Interrupt the RoboTwin Codex planner after this many consecutive "
            "responses without environment progress."
        ),
    )
    parser.add_argument(
        "--robotwin-parity-debug",
        action="store_true",
        help=(
            "Capture evaluator-only native traces after the Agent exits. "
            "Never expose these traces to Agent tools."
        ),
    )


def _parse_config(args: argparse.Namespace) -> RunConfig:
    if getattr(args, "dashboard", False):
        raise ValueError(
            "RoboTwin does not yet support the dashboard launcher; run without "
            "--dashboard so --task-name and the parity settings are explicit"
        )
    if not args.task_name:
        raise ValueError("--task-name is required")
    planner_guard_warn_after = int(
        getattr(args, "planner_guard_warn_after", 3)
    )
    planner_guard_abort_after = int(
        getattr(args, "planner_guard_abort_after", 5)
    )
    if planner_guard_warn_after < 1:
        raise ValueError("--planner-guard-warn-after must be at least 1")
    if planner_guard_abort_after <= planner_guard_warn_after:
        raise ValueError(
            "--planner-guard-abort-after must be greater than "
            "--planner-guard-warn-after"
        )
    if args.lingbot_model_revision != CHECKPOINT_REVISION:
        raise ValueError(
            "LingBot model revision must match the parity manifest: "
            f"{CHECKPOINT_REVISION}"
        )
    env_cuda_device, vla_cuda_device = _resolve_cuda_devices(args)
    if args.robotwin_parity_debug and args.vla_endpoint is not None:
        raise ValueError(
            "--robotwin-parity-debug requires RPent to launch and verify the "
            "pinned local LingBot checkpoint; an external --vla-endpoint does "
            "not expose sufficient model identity metadata"
        )
    if args.env_endpoint is None and not args.robotwin_root:
        raise ValueError(
            "--robotwin-root or ROBOTWIN_ROOT is required when launching the env"
        )
    if not args.vla_root:
        raise ValueError(
            "--vla-root or VLA_ROOT is required for the official LingBot client"
        )
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = (
            get_repo_root()
            / "logs"
            / f"{timestamp}_robotwin_{args.task_name}_s{args.seed}"
        )
    output_dir = Path(output_dir)
    recipe_tag = f"robotwin_{args.task_name}_s{args.seed}"
    task_config = getattr(args, "task_config", "demo_randomized")
    seed_mode = getattr(args, "seed_mode", "exact")
    allow_infeasible = bool(getattr(args, "allow_infeasible", True))
    instruction = getattr(args, "instruction", None)
    initial_seed = _initial_native_seed(args.seed, seed_mode)
    workspace, hybrid_context = _resolve_hybrid_context(
        task_name=args.task_name,
        workspace_value=getattr(args, "robotwin_hybrid_workspace", None),
        robotwin_root=getattr(args, "robotwin_root", None),
        instruction=instruction,
    )
    hybrid_context_sha256 = hashlib.sha256(
        hybrid_context.encode("utf-8")
    ).hexdigest()
    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars={
            "task_name": args.task_name,
            "seed": args.seed,
            "initial_seed": initial_seed,
            "seed_mode": seed_mode,
            "task_config": task_config,
            "allow_infeasible": allow_infeasible,
            "instruction": instruction or "<native task_language from state_00>",
            "hybrid_context": hybrid_context,
        },
        dashboard_state=None,
        task_desc={
            "env": "robotwin",
            "task_name": args.task_name,
            "requested_seed": args.seed,
            "initial_native_seed": initial_seed,
            "seed_mode": seed_mode,
            "task_config": task_config,
            "allow_infeasible": allow_infeasible,
            "instruction": instruction,
            "hybrid_workspace": workspace,
            "semantic_recipe_context": hybrid_context,
            "semantic_recipe_context_sha256": hybrid_context_sha256,
            "checkpoint_revision": CHECKPOINT_REVISION,
            "planner_guard_warn_after": planner_guard_warn_after,
            "planner_guard_abort_after": planner_guard_abort_after,
            "env_cuda_device": env_cuda_device,
            "vla_cuda_device": vla_cuda_device,
        },
    )


def _rpc_client(endpoint: str):
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint
    from rpent.utils.socket_rpc import SocketRpcClient

    protocol, host, port = parse_endpoint(endpoint)
    if protocol == "http":
        return HttpRpcClient(f"http://{host}:{port}")
    if protocol == "socket":
        return SocketRpcClient(host, port)
    raise ValueError(f"unsupported RPC protocol: {protocol!r}")


def _wait_for_tcp(host: str, port: int, daemon, timeout_s: float = 900.0) -> None:
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if daemon is not None and daemon.poll() is not None:
            raise RuntimeError(
                f"{daemon.name} exited before listening; inspect its log"
            )
        try:
            with socket.create_connection((host, port), timeout=1.0):
                return
        except OSError as error:
            last_error = error
            time.sleep(0.5)
    raise TimeoutError(f"LingBot server not ready: {last_error}")


def _parse_vla_endpoint(endpoint: str) -> tuple[str, int]:
    value = endpoint.split("://", 1)[-1]
    host, separator, port_text = value.rpartition(":")
    if not separator or not host or not port_text:
        raise ValueError("--vla-endpoint must be [ws://]host:port")
    return host, int(port_text)


def _vla_code_dir(vla_root: Path) -> Path:
    code_dir = vla_root / "code"
    return code_dir if code_dir.is_dir() else vla_root


def _metadata_revisions(model_path: Path) -> set[str]:
    revisions = set()
    metadata_root = model_path / ".cache" / "huggingface" / "download"
    if not metadata_root.is_dir():
        return revisions
    for path in metadata_root.rglob("*.metadata"):
        try:
            first_line = path.read_text().splitlines()[0].strip()
        except (OSError, IndexError, UnicodeDecodeError):
            continue
        if re.fullmatch(r"[0-9a-f]{40}", first_line):
            revisions.add(first_line)
    return revisions


def _verify_local_model_contract(
    model_path: Path, expected_revision: str
) -> dict[str, Any]:
    required = [
        model_path / "config.json",
        model_path / "lingbotvla_cli.yaml",
        model_path / MODEL_MANIFEST["norm_stats"],
        model_path / MODEL_MANIFEST["qwen_base"],
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise ValueError(f"LingBot model snapshot is incomplete: {missing}")

    evidence = None
    resolved = model_path.resolve()
    if expected_revision in resolved.parts:
        evidence = "huggingface_snapshot_path"
    elif _metadata_revisions(model_path) == {expected_revision}:
        evidence = "huggingface_local_dir_metadata"
    elif (model_path / ".git").exists():
        completed = subprocess.run(
            ["git", "-C", str(model_path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stdout.strip() == expected_revision:
            evidence = "git_head"
    if evidence is None:
        raise ValueError(
            "cannot verify LingBot checkpoint revision from the local snapshot; "
            f"expected {expected_revision}"
        )
    return {
        **MODEL_MANIFEST,
        "model_path": str(resolved),
        "revision_evidence": evidence,
    }


def _subprocess_env(cuda_device: str | None, **extra: str) -> dict[str, str]:
    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    env.update(extra)
    return env


def _resolve_cuda_devices(
    args: argparse.Namespace,
) -> tuple[str | None, str | None]:
    shared = getattr(args, "cuda_device", None)
    env_device = getattr(args, "env_cuda_device", None)
    vla_device = getattr(args, "vla_cuda_device", None)
    if shared is not None and (env_device is not None or vla_device is not None):
        raise ValueError(
            "--cuda-device cannot be combined with --env-cuda-device or "
            "--vla-cuda-device"
        )
    if shared is not None:
        value = str(shared)
        return value, value
    return (
        str(env_device) if env_device is not None else None,
        str(vla_device) if vla_device is not None else None,
    )


def _parse_nvidia_smi_gpu_rows(output: str) -> list[dict[str, Any]]:
    rows = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        index, uuid, memory_used_mib, utilization_gpu_percent = parts
        try:
            rows.append(
                {
                    "index": index,
                    "uuid": uuid,
                    "memory_used_mib": int(memory_used_mib),
                    "utilization_gpu_percent": int(utilization_gpu_percent),
                }
            )
        except ValueError:
            continue
    return rows


def _gpu_snapshot() -> list[dict[str, Any]]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    return _parse_nvidia_smi_gpu_rows(completed.stdout)


def _selected_gpu(
    rows: list[dict[str, Any]], device: str | None
) -> dict[str, Any] | None:
    if device is None:
        return None
    for row in rows:
        if device in (row["index"], row["uuid"]):
            return dict(row)
    return {
        "index": device if device.isdigit() else None,
        "uuid": device if device.startswith("GPU-") else None,
        "unresolved": True,
    }


def _write_resource_manifest(
    path: Path,
    *,
    env_device: str | None,
    vla_device: str | None,
    before_start: list[dict[str, Any]],
    after_start: list[dict[str, Any]] | None = None,
) -> None:
    payload = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "env_gpu": _selected_gpu(before_start, env_device),
        "vla_gpu": _selected_gpu(before_start, vla_device),
        "before_start": before_start,
        "after_start": after_start,
    }
    path.write_text(json.dumps(payload, indent=2))


def _init_runtime(
    args: argparse.Namespace, output_dir: Path
) -> tuple[list["ProcessDaemon"], dict[str, Any]]:
    daemons: list["ProcessDaemon"] = []
    try:
        return _init_runtime_impl(args, output_dir, daemons)
    except Exception:
        for daemon in reversed(daemons):
            daemon.stop()
        raise


def _init_runtime_impl(
    args: argparse.Namespace,
    output_dir: Path,
    daemons: list["ProcessDaemon"],
) -> tuple[list["ProcessDaemon"], dict[str, Any]]:
    from robots.robotwin.env_client import RoboTwinEnvClient
    from robots.robotwin.vla_client import LingBotVLAClient
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.rpc import wait_for_ready

    env_cuda_device, vla_cuda_device = _resolve_cuda_devices(args)
    resource_manifest_path = output_dir / "robotwin_resource_manifest.json"
    gpu_before_start = _gpu_snapshot()
    _write_resource_manifest(
        resource_manifest_path,
        env_device=env_cuda_device,
        vla_device=vla_cuda_device,
        before_start=gpu_before_start,
    )
    robotwin_root = (
        str(Path(args.robotwin_root).expanduser().resolve())
        if args.robotwin_root
        else None
    )
    assets_root = (
        str(Path(args.robotwin_assets_root).expanduser().resolve())
        if args.robotwin_assets_root
        else robotwin_root
    )
    curobo_root = (
        str(Path(args.curobo_root).expanduser().resolve())
        if args.curobo_root
        else str(Path(robotwin_root) / "envs" / "curobo")
        if robotwin_root
        else None
    )
    vla_root = Path(args.vla_root).expanduser().resolve()
    initial_seed = _initial_native_seed(args.seed, args.seed_mode)
    vla_code_dir = _vla_code_dir(vla_root)
    if not vla_code_dir.is_dir():
        raise ValueError(f"LingBot source directory not found: {vla_code_dir}")
    if args.vla_endpoint is None:
        if not args.lingbot_model_path:
            raise ValueError(
                "provide --lingbot-model-path when launching the official server"
            )
        model_path = Path(args.lingbot_model_path).expanduser().resolve()
        model_contract = _verify_local_model_contract(
            model_path, args.lingbot_model_revision
        )
    else:
        model_contract = {
            **MODEL_MANIFEST,
            "verification_status": "unverified_external_endpoint",
            "endpoint": args.vla_endpoint,
        }
    (output_dir / "robotwin_model_contract.json").write_text(
        json.dumps(model_contract, indent=2)
    )

    if args.env_endpoint is None:
        if robotwin_root is None:
            raise ValueError("--robotwin-root is required to launch the env server")
        if assets_root is None:
            raise ValueError(
                "--robotwin-assets-root is required to launch the env server"
            )
        if curobo_root is None:
            raise ValueError("--curobo-root is required to launch the env server")
        host, env_port = "127.0.0.1", pick_free_port()
        env_daemon = ProcessDaemon(
            "robotwin_env_server",
            [
                args.env_python,
                str(Path(__file__).with_name("env_server.py")),
                "--task-name",
                args.task_name,
                "--task-config",
                args.task_config,
                "--seed",
                str(initial_seed),
                "--robotwin-root",
                robotwin_root,
                "--assets-root",
                assets_root,
                "--curobo-root",
                curobo_root,
                "--transport",
                "http",
                "--host",
                host,
                "--port",
                str(env_port),
                "--parent-watch",
                *(["--allow-debug"] if args.robotwin_parity_debug else []),
            ],
            env=_subprocess_env(
                env_cuda_device,
                ROBOTWIN_ROOT=robotwin_root,
                ROBOTWIN_CUROBO_ROOT=curobo_root,
                ASSETS_PATH=assets_root,
                GIT_CONFIG_COUNT="1",
                GIT_CONFIG_KEY_0="safe.directory",
                GIT_CONFIG_VALUE_0=curobo_root,
                PYTHONPATH=os.pathsep.join(
                    [
                        robotwin_root,
                        str(Path(curobo_root) / "src"),
                        os.environ.get("PYTHONPATH", ""),
                    ]
                ),
            ),
            log_path=str(output_dir / "robotwin_env_server.log"),
            cwd=robotwin_root,
        )
        env_daemon.start()
        daemons.append(env_daemon)
        env_rpc = _rpc_client(f"http://{host}:{env_port}")
        wait_for_ready(env_rpc, daemon=env_daemon, timeout_s=900)
    else:
        env_rpc = _rpc_client(args.env_endpoint)
        wait_for_ready(env_rpc)

    if args.vla_endpoint is None:
        host, vla_port = "127.0.0.1", pick_free_port()
        norm_path = model_path / "norm_stats" / "robotwin_eef.json"
        qwen_path = model_path / "qwen_base"
        vla_daemon = ProcessDaemon(
            "lingbot_vla_server",
            [
                args.vla_python,
                str(Path(__file__).with_name("vla_server.py")),
                "--model-path",
                str(model_path),
                "--use-length",
                "50",
                "--port",
                str(vla_port),
                "--norm-path",
                str(norm_path),
            ],
            env=_subprocess_env(
                vla_cuda_device,
                QWEN25_PATH=str(qwen_path),
                PYTHONPATH=os.pathsep.join(
                    [
                        str(vla_code_dir),
                        os.environ.get("PYTHONPATH", ""),
                    ]
                ),
            ),
            log_path=str(output_dir / "lingbot_vla_server.log"),
            cwd=str(vla_code_dir),
        )
        vla_daemon.start()
        daemons.append(vla_daemon)
        _wait_for_tcp(host, vla_port, vla_daemon)
    else:
        host, vla_port = _parse_vla_endpoint(args.vla_endpoint)
        _wait_for_tcp(host, vla_port, None)

    _write_resource_manifest(
        resource_manifest_path,
        env_device=env_cuda_device,
        vla_device=vla_cuda_device,
        before_start=gpu_before_start,
        after_start=_gpu_snapshot(),
    )
    env = RoboTwinEnvClient(
        env_rpc,
        expected_contract={
            "contract_version": CONTRACT_VERSION,
            "profile": "downloads_hybrid",
            "compatibility_id": COMPATIBILITY_ID,
            "asset_snapshot": {
                "file_count": 20854,
                "tree_sha256": (
                    "6ef76bdd0b4b8fefbc5d8dc855563e3b4c4c03674c586079141ab2b66079c12b"
                ),
            },
            "action_specs": {
                "qpos": {"layout": "qpos14", "shape": [14]},
                "ee": {
                    "layout": "eef16",
                    "shape": [16],
                    "frame": "world",
                    "position_unit": "metres",
                    "quaternion_order": "wxyz",
                },
            },
            "camera_specs": {
                "policy": {
                    "views": [
                        "cam_high",
                        "cam_left_wrist",
                        "cam_right_wrist",
                    ],
                    "native_resolution": [320, 240],
                    "model_resolution": [224, 224],
                },
                "agent": {
                    "native_views": ["head", "left_wrist", "right_wrist"],
                    "high_resolution": [1024, 1024],
                    "depth_unit": "metres",
                    "world_frame": "world",
                },
            },
            "episode_budget_counter": "take_action_cnt",
            "canonical_success": "TASK_ENV.eval_success",
            "planner_contract": {
                "repository": "https://github.com/NVlabs/curobo.git",
                "revision": "2fbffc35225398cf9d5f382804faa9de2608753b",
                "clean": True,
                "backend": "curobo",
            },
            "mutation_protocol": {
                "scope": "server_instance",
                "guarantee": "at-most-once",
                "query_method": "get_mutation_result",
            },
        },
    )
    (output_dir / "robotwin_env_contract.json").write_text(
        json.dumps(env.capabilities, indent=2)
    )
    native_model_contract = env.capabilities.get("model_contract", {})
    for key in (
        "checkpoint",
        "revision",
        "policy_name",
        "norm_stats",
        "qwen_base",
        "camera_order",
        "state_layout",
        "action_layout",
        "default_use_length",
    ):
        manifest_key = {
            "checkpoint": "repository",
            "default_use_length": "use_length",
        }.get(key, key)
        if native_model_contract.get(key) != MODEL_MANIFEST.get(manifest_key):
            raise RuntimeError(
                f"RoboTwin model contract mismatch for {key}: "
                f"env={native_model_contract.get(key)!r}, "
                f"manifest={MODEL_MANIFEST.get(manifest_key)!r}"
            )
    model = LingBotVLAClient(
        host=host,
        port=vla_port,
        source_code_dir=vla_code_dir,
    )
    return daemons, {
        "env": env,
        "model": model,
        "seed": initial_seed,
        "seed_mode": args.seed_mode,
        "allow_infeasible": args.allow_infeasible,
        "instruction_type": args.instruction_type,
        "instruction": args.instruction,
        "allow_reset": args.allow_reset,
        "capture_debug_on_close": args.robotwin_parity_debug,
    }
