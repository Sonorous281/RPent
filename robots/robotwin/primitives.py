# Copyright 2026 The RPent Authors.

"""Downloads-compatible RoboTwin primitive orchestration."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from robots.robotwin.env_client import RoboTwinEnvClient
from robots.robotwin.vla_client import LingBotVLAClient


def _qmult(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = left
    w2, x2, y2, z2 = right
    return np.asarray(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


class RoboTwinPrimitives:
    """Agent-facing composition over RLinf's native capability contract."""

    def __init__(
        self,
        *,
        env: RoboTwinEnvClient,
        model: LingBotVLAClient,
        seed: int,
        seed_mode: str = "exact",
        allow_infeasible: bool = True,
        instruction_type: str = "seen",
        instruction: str | None = None,
        allow_reset: bool = False,
        capture_debug_on_close: bool = False,
    ):
        if seed_mode not in ("exact", "walk"):
            raise ValueError("seed_mode must be 'exact' or 'walk'")
        self.env = env
        self.model = model
        self.seed = int(seed)
        self.seed_mode = seed_mode
        self.allow_infeasible = bool(allow_infeasible)
        self.instruction_type = instruction_type
        self.instruction = instruction
        self.allow_reset = allow_reset
        self.capture_debug_on_close = capture_debug_on_close
        self._primitive_counter = itertools.count(1)
        self.primitive_calls = 0
        self.policy_actions = 0
        self.native_actions = 0
        self.last_primitive_id: str | None = None

    def _primitive_id(self, name: str) -> str:
        self.primitive_calls += 1
        self.last_primitive_id = f"primitive-{next(self._primitive_counter):06d}-{name}"
        return self.last_primitive_id

    @staticmethod
    def _mutation_result(record: dict[str, Any]) -> dict[str, Any]:
        result = record.get("result")
        if record.get("state") == "completed" and isinstance(result, dict):
            return {**result, "mutation": record}
        return {
            "success": False,
            "error": record.get("error") or record.get("state"),
            "executed_actions": int(record.get("executed_actions", 0)),
            "episode_status": record.get("episode_status"),
            "mutation": record,
        }

    def reset(
        self,
        *,
        instruction: str | None = None,
        feasibility_precheck: bool = True,
    ) -> dict[str, Any]:
        primitive_id = self._primitive_id("reset")
        mutation_id = self.env.new_mutation_id(primitive_id, "reset")
        options: dict[str, Any] = {
            "exact_seed": self.seed_mode == "exact",
            "allow_infeasible": self.allow_infeasible,
            "instruction_type": self.instruction_type,
            "feasibility_precheck": feasibility_precheck,
        }
        resolved_instruction = self.instruction if instruction is None else instruction
        if resolved_instruction is not None:
            options["instruction"] = resolved_instruction
        record = self.env.reset_episode(
            self.seed,
            mutation_id=mutation_id,
            reset_options=options,
        )
        result = self._mutation_result(record)
        result["success"] = record.get("state") == "completed"
        return result

    def lingbot_act(
        self, *, chunks: int = 4, use_length: int = 50, prompt: str | None = None
    ) -> dict[str, Any]:
        if int(use_length) != 50:
            raise ValueError("RoboTwin LingBot model contract requires use_length=50")
        primitive_id = self._primitive_id("lingbot_act")
        executed = 0
        native_prompt = None
        for chunk_idx in range(int(chunks)):
            status = self.env.get_episode_status()
            if status["success"] or status["truncated"]:
                break
            observation = self.env.capture_policy_observation()
            native_prompt = observation["task"]
            actions = self.model.infer(observation)[:50]
            mutation_id = self.env.new_mutation_id(
                primitive_id, f"eef-chunk-{chunk_idx}"
            )
            version = {
                "episode_generation": observation["episode_generation"],
                "mutation_seq": observation["mutation_seq"],
            }
            record = self.env.execute_actions(
                "ee",
                actions,
                mutation_id=mutation_id,
                expected_version=version,
            )
            result = self._mutation_result(record)
            n = int(result.get("executed_actions", 0))
            executed += n
            self.policy_actions += n
            self.native_actions += n
            if record.get("state") != "completed":
                return {
                    **result,
                    "executed_steps": executed,
                    "prompt": native_prompt,
                }
        return {
            "success": True,
            "executed_steps": executed,
            "prompt": native_prompt,
            "agent_prompt_ignored": prompt is not None,
            "ignored_agent_prompt": prompt,
            "episode_status": self.env.get_episode_status(),
        }

    def move_to(
        self,
        *,
        arm: str,
        xyz: list[float],
        quat: list[float] | None = None,
        gripper: float | None = None,
        substeps: int = 25,
        _primitive_name: str = "move_to",
    ) -> dict[str, Any]:
        primitive_id = self._primitive_id(_primitive_name)
        state = self.env.get_robot_state()
        if quat is None:
            key = "left_eef_pose" if arm == "left" else "right_eef_pose"
            quat = np.asarray(state[key], dtype=np.float64)[3:].tolist()
        target = np.asarray([*xyz, *quat], dtype=np.float64)
        planned = self.env.plan_arm_path(arm, target)
        if planned["status"] != "Success" or planned.get("position") is None:
            return {
                "success": False,
                "plan_status": planned["status"],
                "hint": "target may be unreachable or in collision",
            }
        path = np.asarray(planned["position"], dtype=np.float64)
        if substeps > 0 and len(path) > substeps:
            indices = np.linspace(0, len(path) - 1, substeps).astype(int)
            path = path[indices]
        updates = [
            {"arm": arm, "arm_qpos": waypoint, "gripper": gripper} for waypoint in path
        ]
        mutation_id = self.env.new_mutation_id(primitive_id, "qpos-waypoints")
        expected_plan = {
            "episode_generation": planned["episode_generation"],
            "mutation_seq": planned["mutation_seq"],
        }
        record = self.env.execute_qpos_updates(
            updates,
            mutation_id=mutation_id,
            expected_plan=expected_plan,
        )
        result = self._mutation_result(record)
        executed = int(result.get("executed_actions", 0))
        self.native_actions += executed
        if record.get("state") != "completed":
            return {
                **result,
                "plan_status": planned["status"],
                "waypoints": len(path),
                "executed_steps": executed,
            }
        final = self.env.get_robot_state()
        key = "left_eef_pose" if arm == "left" else "right_eef_pose"
        final_pose = np.asarray(final[key], dtype=np.float64)
        return {
            **result,
            "success": record.get("state") == "completed",
            "plan_status": planned["status"],
            "waypoints": len(path),
            "executed_steps": executed,
            "final_eef_xyz": final_pose[:3],
            "final_dist_m": float(
                np.linalg.norm(final_pose[:3] - np.asarray(xyz, dtype=np.float64))
            ),
        }

    def rotate_wrist(
        self,
        *,
        arm: str,
        delta_yaw_deg: float,
        gripper: float | None = None,
        substeps: int = 25,
    ) -> dict[str, Any]:
        state = self.env.get_robot_state()
        key = "left_eef_pose" if arm == "left" else "right_eef_pose"
        pose = np.asarray(state[key], dtype=np.float64)
        yaw = np.deg2rad(float(delta_yaw_deg))
        world_z = np.asarray([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])
        result = self.move_to(
            arm=arm,
            xyz=pose[:3].tolist(),
            quat=_qmult(world_z, pose[3:]).tolist(),
            gripper=gripper,
            substeps=substeps,
            _primitive_name="rotate_wrist",
        )
        result["requested_delta_yaw_deg"] = float(delta_yaw_deg)
        return result

    def set_gripper(
        self,
        *,
        arm: str,
        val: float,
        steps: int = 10,
        _primitive_name: str = "set_gripper",
    ) -> dict[str, Any]:
        primitive_id = self._primitive_id(_primitive_name)
        state = self.env.get_robot_state()
        current = float(state[f"{arm}_gripper"])
        step_count = int(steps)
        target = float(val)
        values = [
            current + (target - current) * index / step_count
            for index in range(1, step_count + 1)
        ]
        updates = [{"arm": arm, "gripper": float(value)} for value in values]
        mutation_id = self.env.new_mutation_id(primitive_id, "gripper")
        version = {
            "episode_generation": state["episode_generation"],
            "mutation_seq": state["mutation_seq"],
        }
        record = self.env.execute_qpos_updates(
            updates,
            mutation_id=mutation_id,
            expected_plan=version,
        )
        result = self._mutation_result(record)
        executed = int(result.get("executed_actions", 0))
        self.native_actions += executed
        if record.get("state") != "completed":
            return {**result, "executed_steps": executed}
        now = self.env.get_robot_state()
        return {
            **result,
            "success": record.get("state") == "completed",
            "gripper_val": float(now[f"{arm}_gripper"]),
            "executed_steps": executed,
        }

    def release(self, *, arm: str, val: float = 1.0, steps: int = 10) -> dict[str, Any]:
        return self.set_gripper(
            arm=arm,
            val=val,
            steps=steps,
            _primitive_name="release",
        )

    def controlled_reset(self) -> dict[str, Any]:
        if not self.allow_reset:
            return {"success": False, "error": "reset is disabled for this run"}
        return self.reset(feasibility_precheck=False)

    def status(self) -> dict[str, Any]:
        status = self.env.get_episode_status()
        return {
            **status,
            "primitive_calls": self.primitive_calls,
            "policy_actions": self.policy_actions,
            "native_actions": self.native_actions,
        }

    def finish(self, *, status: str, summary: str) -> dict[str, Any]:
        native = self.status()
        requested_success = status.lower() == "success"
        state_trustworthy = bool(native.get("state_trustworthy", False))
        verified_success = state_trustworthy and bool(native["success"])
        reported_status = (
            "success"
            if verified_success
            else ("failure" if requested_success else status)
        )
        return {
            "_finish": True,
            "status": reported_status,
            "summary": summary,
            "requested_success": requested_success,
            "success": verified_success,
            "state_trustworthy": state_trustworthy,
            "native_status": native,
        }
