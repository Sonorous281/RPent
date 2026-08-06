# Copyright 2026 The RPent Authors.

"""Thin RPC client for the RLinf RoboTwin hybrid capability surface."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

from rpent.utils.rpc import RpcClient, RpcError

READ_TIMEOUT_S = 120.0
MUTATION_TIMEOUT_S = 600.0
RECOVERY_TIMEOUT_S = 900.0


def _is_uncertain_transport_error(error: Exception) -> bool:
    """Return whether a mutation may have reached the server."""
    if isinstance(error, RpcError):
        return error.server_traceback is None or (
            "TimeoutError" in error.server_traceback
        )
    return isinstance(error, (ConnectionError, EOFError, OSError, TimeoutError))


def _validate_contract_subset(
    actual: dict[str, Any],
    expected: dict[str, Any],
    *,
    path: str = "capabilities",
) -> None:
    for key, expected_value in expected.items():
        field_path = f"{path}.{key}"
        if key not in actual:
            raise RuntimeError(f"RoboTwin capability is missing {field_path}")
        actual_value = actual[key]
        if isinstance(expected_value, dict):
            if not isinstance(actual_value, dict):
                raise RuntimeError(
                    f"RoboTwin capability mismatch for {field_path}: "
                    f"expected mapping, actual={actual_value!r}"
                )
            _validate_contract_subset(
                actual_value,
                expected_value,
                path=field_path,
            )
        elif actual_value != expected_value:
            raise RuntimeError(
                f"RoboTwin capability mismatch for {field_path}: "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )


class RoboTwinEnvClient:
    """Client for one ``RoboTwinEnv(profile='downloads_hybrid')`` instance."""

    def __init__(self, client: RpcClient, *, expected_contract: dict[str, Any]):
        self._client = client
        self._mutation_counter = 0
        self.mutation_observer: Callable[[dict[str, Any]], None] | None = None
        self.capabilities = self._client.call(
            "env.get_capabilities", timeout_s=READ_TIMEOUT_S
        )
        _validate_contract_subset(self.capabilities, expected_contract)
        self.server_instance_id = self.capabilities["server_instance_id"]
        expected_prefix = f"{self.server_instance_id}:"
        if self.capabilities.get("mutation_id_prefix") != expected_prefix:
            raise RuntimeError(
                f"RoboTwin mutation scope mismatch: expected prefix {expected_prefix!r}"
            )

    def new_mutation_id(self, primitive_id: str, operation: str) -> str:
        """Return a business idempotency key for one state-changing operation."""
        self._mutation_counter += 1
        suffix = uuid.uuid4().hex
        return (
            f"{self.server_instance_id}:{primitive_id}:{operation}:"
            f"{self._mutation_counter}:{suffix}"
        )

    def _mutate(
        self,
        method: str,
        *,
        mutation_id: str,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        self._notify_mutation(
            {
                "method": method,
                "mutation_id": mutation_id,
                "phase": "submitted",
            }
        )
        try:
            result = self._client.call(
                f"env.{method}",
                kwargs={**kwargs, "mutation_id": mutation_id},
                timeout_s=MUTATION_TIMEOUT_S,
            )
            self._notify_mutation(
                {
                    "method": method,
                    "mutation_id": mutation_id,
                    "phase": "returned",
                    "state": result.get("state") if isinstance(result, dict) else None,
                }
            )
            return result
        except Exception as original_error:
            if not _is_uncertain_transport_error(original_error):
                raise
            try:
                recovered = self.get_mutation_result(
                    mutation_id, timeout_s=RECOVERY_TIMEOUT_S
                )
            except Exception:
                raise original_error
            if recovered is None:
                raise original_error
            result = {**recovered, "recovered_after_transport_error": True}
            self._notify_mutation(
                {
                    "method": method,
                    "mutation_id": mutation_id,
                    "phase": "recovered",
                    "state": result.get("state"),
                }
            )
            return result

    def _notify_mutation(self, event: dict[str, Any]) -> None:
        observer = self.mutation_observer
        if observer is None:
            return
        try:
            observer(event)
        except Exception:
            # Telemetry must never alter mutation execution or recovery.
            pass

    def get_robot_state(self) -> dict[str, Any]:
        return self._client.call("env.get_robot_state", timeout_s=READ_TIMEOUT_S)

    def capture_policy_observation(self) -> dict[str, Any]:
        return self._client.call(
            "env.capture_policy_observation", timeout_s=READ_TIMEOUT_S
        )

    def capture_agent_observation(self) -> dict[str, Any]:
        return self._client.call(
            "env.capture_agent_observation", timeout_s=READ_TIMEOUT_S
        )

    def capture_debug_state(self) -> dict[str, Any]:
        """Capture evaluator-only state; this method is not registered as a tool."""
        return self._client.call("env.capture_debug_state", timeout_s=READ_TIMEOUT_S)

    def get_episode_status(self) -> dict[str, Any]:
        return self._client.call("env.get_episode_status", timeout_s=READ_TIMEOUT_S)

    def plan_arm_path(self, arm: str, target_pose) -> dict[str, Any]:
        return self._client.call(
            "env.plan_arm_path",
            kwargs={"arm": arm, "target_pose": target_pose},
            timeout_s=READ_TIMEOUT_S,
        )

    def execute_actions(
        self,
        action_type: str,
        actions,
        *,
        mutation_id: str,
        expected_version: dict[str, int] | None,
    ) -> dict[str, Any]:
        return self._mutate(
            "execute_actions",
            mutation_id=mutation_id,
            kwargs={
                "action_type": action_type,
                "actions": actions,
                "expected_version": expected_version,
            },
        )

    def execute_qpos_updates(
        self,
        updates: list[dict[str, Any]],
        *,
        mutation_id: str,
        expected_plan: dict[str, int] | None,
    ) -> dict[str, Any]:
        return self._mutate(
            "execute_qpos_updates",
            mutation_id=mutation_id,
            kwargs={"updates": updates, "expected_plan": expected_plan},
        )

    def reset_episode(
        self,
        seed: int,
        *,
        mutation_id: str,
        reset_options: dict[str, Any],
    ) -> dict[str, Any]:
        return self._mutate(
            "reset_episode",
            mutation_id=mutation_id,
            kwargs={"seed": seed, "reset_options": reset_options},
        )

    def get_mutation_result(
        self, mutation_id: str, *, timeout_s: float = READ_TIMEOUT_S
    ) -> dict[str, Any] | None:
        return self._client.call(
            "env.get_mutation_result",
            kwargs={"mutation_id": mutation_id},
            timeout_s=timeout_s,
        )
