# Copyright 2026 The RPent Authors.

"""Thin launcher for the official LingBot-VLA WebSocket server."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any


def _build_policy(
    policy_cls,
    *,
    model_path: str,
    norm_path: str,
    use_length: int,
    num_denoising_step: int,
    use_compile: bool,
) -> Any:
    return policy_cls(
        model_path,
        use_length=use_length,
        robot_norm_path=norm_path,
        num_denoising_step=num_denoising_step,
        use_compile=use_compile,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start the official LingBot-VLA WebSocket policy server"
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--norm-path", required=True)
    parser.add_argument("--use-length", type=int, default=50)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--num-denoising-step", type=int, default=10)
    parser.add_argument("--use-compile", action="store_true")
    parser.add_argument(
        "--runtime-config-root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help=(
            "Directory containing configs/robot_configs/robotwin_eef.yaml. "
            "The official server resolves this path relative to its cwd."
        ),
    )
    args = parser.parse_args()

    runtime_config_root = args.runtime_config_root.expanduser().resolve()
    robot_config = (
        runtime_config_root / "configs" / "robot_configs" / "robotwin_eef.yaml"
    )
    if not robot_config.is_file():
        raise FileNotFoundError(f"LingBot robot config not found: {robot_config}")
    os.chdir(runtime_config_root)

    from deploy.lingbot_vla_policy import LingbotVLAServer
    from deploy.websocket_policy_server import WebsocketPolicyServer

    policy = _build_policy(
        LingbotVLAServer,
        model_path=args.model_path,
        norm_path=args.norm_path,
        use_length=args.use_length,
        num_denoising_step=args.num_denoising_step,
        use_compile=args.use_compile,
    )
    WebsocketPolicyServer(policy, port=args.port).serve_forever()


if __name__ == "__main__":
    main()
