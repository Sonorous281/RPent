# Copyright 2026 The RPent Authors.

"""RoboTwin tool schemas and observation artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _tool_error(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {
        "success": False,
        "error": {"code": code, "message": message, **details},
    }


def _load_world_xyz(
    output_dir: Path,
    *,
    view: str,
    step: int | None,
) -> tuple[dict[str, Any] | None, np.ndarray | None, dict[str, Any] | None]:
    """Load one persisted agent-visible world map without touching the env."""
    output_dir = output_dir.resolve()
    state_path = (
        output_dir / "latest_state.json"
        if step is None
        else output_dir / f"state_{int(step):02d}.json"
    )
    if not state_path.is_file():
        return (
            None,
            None,
            _tool_error(
                "state_not_found",
                "The requested RoboTwin state artifact does not exist.",
                step=step,
            ),
        )
    try:
        state = json.loads(state_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        return (
            None,
            None,
            _tool_error(
                "state_invalid",
                "The requested RoboTwin state artifact cannot be read.",
                detail=str(error),
            ),
        )
    actual_step = state.get("step_idx")
    if step is not None and actual_step != int(step):
        return (
            None,
            None,
            _tool_error(
                "step_mismatch",
                "The state artifact does not match the requested step.",
                requested_step=int(step),
                actual_step=actual_step,
            ),
        )
    views = state.get("artifacts")
    if not isinstance(views, dict) or view not in views:
        return (
            None,
            None,
            _tool_error(
                "view_not_found",
                "The requested view is unavailable in this state.",
                view=view,
                available_views=sorted(views) if isinstance(views, dict) else [],
            ),
        )
    artifact = views[view]
    world_path_value = artifact.get("world_xyz") if isinstance(artifact, dict) else None
    if not world_path_value:
        return (
            None,
            None,
            _tool_error(
                "world_xyz_not_found",
                "The requested view has no persisted world map.",
                view=view,
                step_idx=actual_step,
            ),
        )
    world_path = Path(world_path_value).expanduser().resolve()
    if not world_path.is_relative_to(output_dir):
        return (
            None,
            None,
            _tool_error(
                "artifact_outside_run",
                "The world-map artifact is outside the current run directory.",
                view=view,
            ),
        )
    expected_tag = f"_{int(actual_step):02d}.npy"
    if actual_step is None or not world_path.name.endswith(expected_tag):
        return (
            None,
            None,
            _tool_error(
                "frame_mismatch",
                "The world-map artifact does not belong to the requested state.",
                view=view,
                step_idx=actual_step,
                artifact=world_path.name,
            ),
        )
    if not world_path.is_file():
        return (
            None,
            None,
            _tool_error(
                "world_xyz_not_found",
                "The persisted world-map artifact does not exist.",
                view=view,
                step_idx=actual_step,
            ),
        )
    try:
        world = np.load(world_path, allow_pickle=False)
    except (OSError, ValueError) as error:
        return (
            None,
            None,
            _tool_error(
                "world_xyz_invalid",
                "The persisted world map cannot be read.",
                detail=str(error),
            ),
        )
    if world.ndim != 3 or world.shape[2] != 3:
        return (
            None,
            None,
            _tool_error(
                "world_xyz_shape",
                "A RoboTwin world map must have shape [H,W,3].",
                actual_shape=list(world.shape),
            ),
        )
    return state, np.asarray(world), None


def sample_world_xyz(
    output_dir: Path,
    *,
    view: str,
    pixels: list[list[int]],
    step: int | None = None,
    neighborhood: int = 1,
) -> dict[str, Any]:
    """Return deterministic median world coordinates around image pixels."""
    state, world, error = _load_world_xyz(output_dir, view=view, step=step)
    if error is not None:
        return error
    assert state is not None and world is not None
    radius = int(neighborhood)
    if radius < 0 or radius > 32:
        return _tool_error(
            "invalid_neighborhood",
            "neighborhood must be an integer from 0 through 32.",
        )
    if not isinstance(pixels, list) or not pixels or len(pixels) > 256:
        return _tool_error(
            "invalid_pixels",
            "pixels must contain between 1 and 256 [row,col] pairs.",
        )
    height, width = world.shape[:2]
    samples: list[dict[str, Any]] = []
    for pixel in pixels:
        if (
            not isinstance(pixel, (list, tuple))
            or len(pixel) != 2
            or not all(isinstance(value, (int, np.integer)) for value in pixel)
        ):
            return _tool_error(
                "invalid_pixel",
                "Every pixel must be an integer [row,col] pair.",
                pixel=pixel,
            )
        row, col = (int(pixel[0]), int(pixel[1]))
        if row < 0 or row >= height or col < 0 or col >= width:
            return _tool_error(
                "pixel_out_of_bounds",
                "The pixel is outside this view's world map. Use the exact "
                "artifact view whose RGB supplied the pixel; do not reuse "
                "high-resolution pixels with a base-resolution view.",
                pixel=[row, col],
                shape=[height, width],
                view=view,
                coordinate_space=view,
                valid_row_range=[0, height - 1],
                valid_col_range=[0, width - 1],
            )
        row_start = max(0, row - radius)
        row_end = min(height, row + radius + 1)
        col_start = max(0, col - radius)
        col_end = min(width, col + radius + 1)
        region = world[row_start:row_end, col_start:col_end].reshape(-1, 3)
        finite_counts = np.isfinite(region).sum(axis=0)
        if np.any(finite_counts == 0):
            return _tool_error(
                "no_valid_world_points",
                "The requested pixel neighborhood has no finite xyz coordinate.",
                pixel=[row, col],
                neighborhood=radius,
            )
        xyz = np.nanmedian(region, axis=0)
        samples.append(
            {
                "pixel": [row, col],
                "valid": True,
                "xyz": xyz.tolist(),
                "valid_points": int(np.isfinite(region).all(axis=1).sum()),
                "valid_coordinates": finite_counts.tolist(),
            }
        )
    return {
        "success": True,
        "step_idx": state["step_idx"],
        "frame_id": state["frame_id"],
        "view": view,
        "coordinate_space": view,
        "image_shape": [height, width],
        "pixel_order": "row_col",
        "coordinate_order": "xyz",
        "frame": "world",
        "unit": "metre",
        "neighborhood": radius,
        "samples": samples,
    }


def query_world_map(
    output_dir: Path,
    *,
    view: str,
    bbox: list[int],
    step: int | None = None,
    max_points: int = 256,
) -> dict[str, Any]:
    """Return deterministic row-major samples and statistics for one bbox."""
    state, world, error = _load_world_xyz(output_dir, view=view, step=step)
    if error is not None:
        return error
    assert state is not None and world is not None
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or not all(isinstance(value, (int, np.integer)) for value in bbox)
    ):
        return _tool_error(
            "invalid_bbox",
            "bbox must be [row_start,col_start,row_end,col_end].",
        )
    row_start, col_start, row_end, col_end = map(int, bbox)
    height, width = world.shape[:2]
    if not (0 <= row_start < row_end <= height and 0 <= col_start < col_end <= width):
        return _tool_error(
            "bbox_out_of_bounds",
            "bbox must be a non-empty half-open region inside this view's "
            "world map. Use the exact artifact view whose RGB supplied the "
            "bbox coordinates.",
            bbox=list(map(int, bbox)),
            shape=[height, width],
            view=view,
            coordinate_space=view,
            valid_bbox=[0, 0, height, width],
        )
    limit = int(max_points)
    if limit < 1 or limit > 4096:
        return _tool_error(
            "invalid_max_points",
            "max_points must be an integer from 1 through 4096.",
        )
    region = world[row_start:row_end, col_start:col_end]
    valid_mask = np.isfinite(region).all(axis=2)
    local_rows, local_cols = np.nonzero(valid_mask)
    if not len(local_rows):
        return _tool_error(
            "no_valid_world_points",
            "The requested region contains no finite world coordinates.",
            bbox=list(map(int, bbox)),
        )
    xyz = region[local_rows, local_cols]
    if len(xyz) > limit:
        indices = np.linspace(0, len(xyz) - 1, limit).astype(int)
    else:
        indices = np.arange(len(xyz))
    points = [
        {
            "pixel": [
                int(row_start + local_rows[index]),
                int(col_start + local_cols[index]),
            ],
            "xyz": xyz[index].tolist(),
        }
        for index in indices
    ]
    return {
        "success": True,
        "step_idx": state["step_idx"],
        "frame_id": state["frame_id"],
        "view": view,
        "coordinate_space": view,
        "image_shape": [height, width],
        "bbox": [row_start, col_start, row_end, col_end],
        "bbox_interval": "half_open",
        "pixel_order": "row_col",
        "coordinate_order": "xyz",
        "frame": "world",
        "unit": "metre",
        "valid_points": int(len(xyz)),
        "returned_points": len(points),
        "xyz_min": np.min(xyz, axis=0).tolist(),
        "xyz_max": np.max(xyz, axis=0).tolist(),
        "xyz_median": np.median(xyz, axis=0).tolist(),
        "points": points,
    }


def dump_observation(
    observation: dict[str, Any],
    *,
    output_dir: Path,
    step_idx: int,
    status: dict[str, Any],
    log: dict[str, Any] | None,
) -> dict[str, Any]:
    """Persist one agent-visible observation without simulator oracle state."""
    import imageio.v2 as imageio

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{step_idx:02d}"
    paths: dict[str, dict[str, str]] = {}
    view_specs: dict[str, dict[str, Any]] = {}
    for view_name, view in observation["views"].items():
        view_dir = output_dir / view_name
        view_dir.mkdir(parents=True, exist_ok=True)
        view_paths: dict[str, str] = {}
        if "rgb" in view:
            path = view_dir / f"rgb_{tag}.png"
            imageio.imwrite(path, np.asarray(view["rgb"], dtype=np.uint8))
            view_paths["rgb"] = str(path)
        for field in ("depth", "world_xyz"):
            if field in view:
                path = view_dir / f"{field}_{tag}.npy"
                np.save(path, np.asarray(view[field]))
                view_paths[field] = str(path)
        if "camera_meta" in view:
            path = view_dir / f"camera_meta_{tag}.json"
            path.write_text(
                json.dumps(view["camera_meta"], indent=2, default=_json_default)
            )
            view_paths["camera_meta"] = str(path)
        paths[view_name] = view_paths
        shape_source = next(
            (
                np.asarray(view[field])
                for field in ("rgb", "world_xyz", "depth")
                if field in view
            ),
            None,
        )
        if shape_source is not None and shape_source.ndim >= 2:
            view_specs[view_name] = {
                "coordinate_space": view_name,
                "image_shape": [
                    int(shape_source.shape[0]),
                    int(shape_source.shape[1]),
                ],
                "pixel_order": "row_col",
            }

    state = {
        "step_idx": step_idx,
        "frame_id": observation["frame_id"],
        "task_name": observation["task_name"],
        "task_language": observation["task_language"],
        "object_names": observation["object_names"],
        "robot_state": observation["robot_state"],
        "episode_status": status,
        "artifacts": paths,
        "view_specs": view_specs,
        "log": log,
    }
    state_path = output_dir / f"state_{tag}.json"
    state_path.write_text(json.dumps(state, indent=2, default=_json_default))
    (output_dir / "latest_state.json").write_text(
        json.dumps(state, indent=2, default=_json_default)
    )
    return state


def dump_untrusted_status(
    *,
    output_dir: Path,
    step_idx: int,
    status: dict[str, Any],
    log: dict[str, Any] | None,
) -> dict[str, Any]:
    """Persist status without touching cameras after native state becomes unknown."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{step_idx:02d}"
    state = {
        "step_idx": step_idx,
        "frame_id": None,
        "episode_status": status,
        "artifacts": {},
        "log": log,
        "observation_unavailable": "state_unknown",
    }
    serialized = json.dumps(state, indent=2, default=_json_default)
    (output_dir / f"state_{tag}.json").write_text(serialized)
    (output_dir / "latest_state.json").write_text(serialized)
    return state


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def view_state(output_dir: Path, step_idx: int | None = None) -> dict[str, Any]:
    path = (
        output_dir / "latest_state.json"
        if step_idx is None
        else output_dir / f"state_{step_idx:02d}.json"
    )
    if not path.exists():
        return {"error": f"state artifact not found: {path}"}
    result = json.loads(path.read_text())
    preferred = result.get("artifacts", {}).get("head_hi", {}).get("rgb") or result.get(
        "artifacts", {}
    ).get("head", {}).get("rgb")
    if preferred and Path(preferred).exists():
        result["_image_bytes"] = Path(preferred).read_bytes()
    return result


TOOLS_SPEC = [
    {
        "name": "view_driver_state",
        "description": "Read the latest RoboTwin state and agent-visible camera paths.",
        "input_schema": {
            "type": "object",
            "properties": {"step": {"type": ["integer", "null"]}},
        },
    },
    {
        "name": "render",
        "description": "Capture a fresh synchronized RoboTwin agent observation.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "sample_world_xyz",
        "description": (
            "Read persisted same-frame world xyz around [row,col] pixels. "
            "The view is also the pixel coordinate space: use the exact view "
            "whose RGB supplied the pixels (for example, head_hi pixels require "
            "view=head_hi, never view=head). The current state's view_specs "
            "gives each view's [height,width]. This is read-only and does not "
            "render or move the robot."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "description": (
                        "Artifact view and pixel coordinate space. It must match "
                        "the RGB image used to choose pixels."
                    ),
                },
                "pixels": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 256,
                    "items": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                },
                "step": {"type": ["integer", "null"]},
                "neighborhood": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 32,
                    "default": 1,
                },
            },
            "required": ["view", "pixels"],
        },
    },
    {
        "name": "query_world_map",
        "description": (
            "Read deterministic world-xyz samples from a half-open "
            "[row_start,col_start,row_end,col_end] region. The view is also "
            "the bbox coordinate space and must match the source RGB artifact; "
            "view_specs gives [height,width]. This is read-only."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "view": {
                    "type": "string",
                    "description": (
                        "Artifact view and bbox coordinate space. It must match "
                        "the RGB image used to choose the bbox."
                    ),
                },
                "bbox": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "step": {"type": ["integer", "null"]},
                "max_points": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 4096,
                    "default": 256,
                },
            },
            "required": ["view", "bbox"],
        },
    },
    {
        "name": "lingbot_act",
        "description": (
            "Run LingBot-VLA eef16 actions using the native task instruction. "
            "The optional prompt is recorded but never sent to the policy."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chunks": {"type": "integer", "default": 4},
                "use_length": {"type": "integer", "const": 50, "default": 50},
                "prompt": {"type": ["string", "null"]},
            },
        },
    },
    {
        "name": "move_to",
        "description": (
            "Plan and move one arm to a world-frame xyz and wxyz orientation. "
            "The native planner returns qpos waypoints executed with fresh state."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "xyz": {
                    "type": "array",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "quat": {
                    "type": ["array", "null"],
                    "items": {"type": "number"},
                    "minItems": 4,
                    "maxItems": 4,
                },
                "gripper": {"type": ["number", "null"]},
                "substeps": {"type": "integer", "default": 25},
            },
            "required": ["arm", "xyz"],
        },
    },
    {
        "name": "rotate_wrist",
        "description": "Rotate one EEF about world Z by a relative angle in degrees.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "delta_yaw_deg": {"type": "number"},
                "gripper": {"type": ["number", "null"]},
                "substeps": {"type": "integer", "default": 25},
            },
            "required": ["arm", "delta_yaw_deg"],
        },
    },
    {
        "name": "set_gripper",
        "description": "Linearly move one normalized gripper to val over 10 actions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "val": {"type": "number", "minimum": 0, "maximum": 1},
                "steps": {"type": "integer", "default": 10},
            },
            "required": ["arm", "val"],
        },
    },
    {
        "name": "release",
        "description": "Open one gripper to 1.0 over 10 native actions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "arm": {"type": "string", "enum": ["left", "right"]},
                "val": {"type": "number", "default": 1.0},
                "steps": {"type": "integer", "default": 10},
            },
            "required": ["arm"],
        },
    },
    {
        "name": "reset",
        "description": (
            "Reset the same seed only when this run explicitly allows reset."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "finish",
        "description": (
            "Stop the run. A fresh native status query is authoritative; requesting "
            "success cannot override TASK_ENV.eval_success."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["status", "summary"],
        },
    },
    {
        "name": "quit",
        "description": "Alias for finish with a fresh native status check.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "summary": {"type": "string"},
            },
            "required": ["status", "summary"],
        },
    },
]
