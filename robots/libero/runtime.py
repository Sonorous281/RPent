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

"""Environment startup shared by LIBERO planner workflows."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from robots.libero.suites import resolve_libero_type
from rpent.robots.runtime import start_rpc_server
from rpent.utils.config import get_repo_root, resolve_cuda_device
from rpent.utils.daemon import ProcessDaemon
from rpent.utils.rpc import make_rpc_client

if TYPE_CHECKING:
    from rpent.utils.rpc import RpcClient


def spawn_env_server(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[ProcessDaemon | None, RpcClient]:
    """Connect to an external environment or start an owned LIBERO server."""
    if args.env_endpoint is not None:
        return None, make_rpc_client(args.env_endpoint)

    device = resolve_cuda_device(args, "env")

    log = output_dir / "env_server.log"
    daemon = ProcessDaemon(
        name="env_server",
        cmd=[
            sys.executable,
            str(get_repo_root() / "robots" / "libero" / "env_server.py"),
            "--suite",
            args.suite,
            "--task",
            str(args.task),
            "--seed",
            str(args.seed),
            "--max-episode-steps",
            str(args.max_episode_steps),
            "--transport",
            "http",
            "--host",
            "127.0.0.1",
            # Concurrent episodes start many env servers; each binds its own port.
            "--port",
            "0",
            "--parent-watch",
            *(["--cuda-device", str(device)] if device is not None else []),
        ],
        env_overrides={
            "LIBERO_TYPE": resolve_libero_type(args.suite, args.libero_type),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
            "ROBOT_PLATFORM": "LIBERO",
        },
        log_path=str(log),
    )
    return start_rpc_server(daemon)
