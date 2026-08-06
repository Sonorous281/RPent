# Copyright 2026 The RPent Authors.

"""LingBot-VLA websocket client using the checkpoint's official runtime."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


class LingBotVLAClient:
    """Wrap ``deploy.websocket_client_policy.WebsocketClientPolicy``."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        source_code_dir: str | Path | None = None,
    ):
        self._host = host
        self._port = int(port)
        self._source_code_dir = (
            Path(source_code_dir).expanduser().resolve()
            if source_code_dir is not None
            else None
        )
        self._policy = None

    def _get_policy(self):
        if self._policy is None:
            if self._source_code_dir is not None:
                source = str(self._source_code_dir)
                if source not in sys.path:
                    sys.path.insert(0, source)
            for name in ("https_proxy", "http_proxy", "HTTPS_PROXY", "HTTP_PROXY"):
                os.environ.pop(name, None)
            from deploy.websocket_client_policy import WebsocketClientPolicy

            self._policy = WebsocketClientPolicy(host=self._host, port=self._port)
            self._policy.reset(robo_name="robotwin_eef")
        return self._policy

    def infer(self, observation: dict[str, Any]) -> np.ndarray:
        """Infer one eef16 chunk without changing the native environment."""
        payload = {
            "observation.images.cam_high": observation["images"]["cam_high"],
            "observation.images.cam_left_wrist": observation["images"][
                "cam_left_wrist"
            ],
            "observation.images.cam_right_wrist": observation["images"][
                "cam_right_wrist"
            ],
            "observation.state": np.asarray(observation["state"], dtype=np.float32),
            "task": observation["task"],
        }
        actions = np.asarray(
            self._get_policy().infer(payload)["action"], dtype=np.float64
        )
        if actions.ndim != 2 or actions.shape[1] != 16:
            raise RuntimeError(
                f"LingBot returned {actions.shape}; expected [chunk, 16]"
            )
        return actions
