# Copyright 2026 The RPent Authors.

"""RPC server owning one RLinf RoboTwin hybrid environment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

_RPENT_SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(_RPENT_SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(_RPENT_SOURCE_ROOT))

from rpent.utils.config import get_repo_root, get_rlinf_repo_path  # noqa: E402
from rpent.utils.logging import get_logger  # noqa: E402
from rpent.utils.rpc import RpcFacade  # noqa: E402

logger = get_logger("robotwin_env_server")

RPENT_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (RPENT_ROOT.parent / "RLinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
from rlinf.envs.robotwin.robotwin_env import RoboTwinEnv  # noqa: E402


def build_env_cfg(
    *,
    task_name: str,
    task_config: str,
    seed: int,
    robotwin_root: str,
    assets_root: str,
    allow_debug: bool = False,
) -> Any:
    """Build a single-env RLinf config from the native RoboTwin task YAML."""
    config_path = (
        Path(robotwin_root).expanduser().resolve()
        / "task_config"
        / f"{task_config}.yml"
    )
    if not config_path.is_file():
        raise ValueError(f"RoboTwin task config not found: {config_path}")
    native_task_config = OmegaConf.load(config_path)
    native_task_config.task_name = task_name
    native_task_config.task_config = task_config
    native_task_config.ckpt_setting = "hybrid_lingbot"
    native_task_config.policy_name = "hybrid_lingbot"
    native_task_config.planner_backend = "curobo"
    native_task_config.eval_video_log = False
    native_task_config.render_freq = 0

    return OmegaConf.create(
        {
            "env_type": "robotwin",
            "profile": "downloads_hybrid",
            "allow_hybrid_debug": allow_debug,
            "hybrid_initial_seed": seed,
            "auto_reset": False,
            "ignore_terminations": False,
            "reward_coef": 1.0,
            "use_custom_reward": True,
            "use_rel_reward": True,
            "center_crop": False,
            "seed": seed,
            "group_size": 1,
            "use_fixed_reset_state_ids": True,
            "max_steps_per_rollout_epoch": 450,
            "max_episode_steps": 450,
            "is_eval": True,
            "assets_path": assets_root,
            "seeds_path": None,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": None,
            },
            "enable_offload": False,
            "task_config": native_task_config,
        }
    )


def make_env(
    task_name: str,
    task_config: str,
    seed: int,
    robotwin_root: str,
    assets_root: str,
    curobo_root: str,
    *,
    allow_debug: bool = False,
) -> RoboTwinEnv:
    """Construct the only simulator owner used by an RPent run."""
    if robotwin_root not in sys.path:
        sys.path.insert(0, robotwin_root)
    curobo_source_root = str(Path(curobo_root).expanduser().resolve() / "src")
    if curobo_source_root not in sys.path:
        sys.path.insert(0, curobo_source_root)
    os.environ["ROBOTWIN_CUROBO_ROOT"] = str(Path(curobo_root).expanduser().resolve())
    assets_path = Path(assets_root)
    required_assets = (
        assets_path / "assets" / "embodiments" / "aloha-agilex",
        assets_path / "assets" / "objects",
    )
    missing_assets = [str(path) for path in required_assets if not path.is_dir()]
    if missing_assets:
        raise ValueError(
            "RoboTwin asset snapshot is incomplete; missing directories: "
            f"{missing_assets}"
        )
    cfg = build_env_cfg(
        task_name=task_name,
        task_config=task_config,
        seed=seed,
        robotwin_root=robotwin_root,
        assets_root=str(assets_path),
        allow_debug=allow_debug,
    )
    return RoboTwinEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=False,
    )


def _to_numpy_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {key: _to_numpy_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_numpy_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_numpy_tree(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


class RoboTwinEnvFacade(RpcFacade):
    """Transport-only facade over RLinf's public RoboTwin capabilities."""

    def __init__(self, env: RoboTwinEnv):
        super().__init__()
        self._env = env

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        if not method.startswith("env."):
            raise ValueError(f"unknown RPC method: {method!r}")
        name = method[len("env.") :]
        allowed = {
            "get_capabilities",
            "get_robot_state",
            "capture_policy_observation",
            "capture_agent_observation",
            "capture_debug_state",
            "get_episode_status",
            "plan_arm_path",
            "execute_actions",
            "execute_qpos_updates",
            "reset_episode",
            "get_mutation_result",
        }
        if name not in allowed:
            raise ValueError(f"unknown RoboTwin env method: {name!r}")
        capability = getattr(self._env, name)
        if name != "get_capabilities":
            kwargs = {"env_id": 0, **kwargs}
        return _to_numpy_tree(capability(*args, **kwargs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--task-name", required=True)
    parser.add_argument(
        "--task-config",
        choices=("demo_clean", "demo_randomized"),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--robotwin-root", required=True)
    parser.add_argument("--assets-root", required=True)
    parser.add_argument("--curobo-root", required=True)
    parser.add_argument("--allow-debug", action="store_true")
    parser.add_argument("--parent-watch", action="store_true")
    args = parser.parse_args()

    os.environ["ASSETS_PATH"] = args.assets_root
    env = make_env(
        args.task_name,
        args.task_config,
        args.seed,
        args.robotwin_root,
        args.assets_root,
        args.curobo_root,
        allow_debug=args.allow_debug,
    )
    facade = RoboTwinEnvFacade(env)
    try:
        facade.serve(
            transport=args.transport,
            host=args.host,
            port=args.port,
            parent_watch=args.parent_watch,
        )
    finally:
        env.offload(clear_cache=True)


if __name__ == "__main__":
    main()
