LIBERO
======

`LIBERO <https://libero-project.github.io/>`_ is the default RPent
environment: a MuJoCo/robosuite-based tabletop manipulation benchmark
with four suites (``libero_object``, ``libero_goal``, ``libero_spatial``,
``libero_10``) and three variants (``standard``, ``pro``, ``plus``).
The default VLA is **Pi0.5**, served over HTTP by
``robots/libero/vla_server.py``.

VLA configuration
-----------------

Pi0.5 needs one thing: a checkpoint on disk. Point at it via
``PI05_CHECKPOINT_PATH``:

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

Download the recommended SFT checkpoint from HuggingFace:
`RLinf-Pi05-LIBERO-130-fullshot-SFT
<https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_.

Task selection
--------------

Every LIBERO run picks:

- ``--suite`` — one of the four suites, each optionally suffixed with
  the variant flavor (see below). Examples:
  ``libero_object_task``, ``libero_object_swap``,
  ``libero_goal_lan``, ``libero_spatial_task``,
  ``libero_10_swap``.
- ``--task`` — the task index within the suite.
- ``--seed`` — the environment seed.
- ``--libero-type`` — the LIBERO variant: ``standard`` | ``pro`` |
  ``plus``. If omitted, RPent falls back to ``LIBERO_TYPE`` in the
  environment (default ``pro``).

Suite × variant matrix
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Suite
     - Variants
     - Purpose
   * - ``libero_object``
     - ``_task`` / ``_swap`` / ``_lan``
     - Object-centric tasks with optional target swap or language
       perturbations.
   * - ``libero_goal``
     - ``_task`` / ``_swap`` / ``_lan``
     - Goal-conditioned tasks with optional swap / language
       perturbations.
   * - ``libero_spatial``
     - ``_task`` / ``_lan``
     - Spatial-relations tasks.
   * - ``libero_10``
     - ``_task`` / ``_swap`` / ``_lan``
     - The long-horizon LIBERO-10 suite.

Minimal command
---------------

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0

   rpent --env libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 \
     --max-tokens 8192

What runs where
---------------

- **env_server** (``robots/libero/env_server.py``) — owns the LIBERO
  MuJoCo env and EGL rendering. Exposes ``reset``, ``step``,
  ``chunk_step``, ``render_camera(camera_name="agentview")``, ``get_camera_meta``,
  ``cached_image``, … over an RPC transport (HTTP by default; a
  pickle-framed socket via ``--transport socket``).
- **vla_server** (``robots/libero/vla_server.py``) — owns the Pi0.5
  weights. Exposes ``predict`` over the same RPC transport (HTTP or
  socket).
- **Toolkit** (``robots/libero/toolkit.py``) — defines the tools the
  LLM can call: ``pi0_pick`` (fed to Pi0.5), ``move_to``,
  ``rotate_wrist``, ``back_project``, ``view_driver_state``,
  ``finish``, …

Tools the planner sees
----------------------

By default the LIBERO toolkit exposes:

- ``pi0_pick(prompt)`` — invoke Pi0.5 for a pick chunk driven by
  ``prompt`` (a natural-language pick instruction).
- ``pi0_doubled(prompt)`` — invoke Pi0.5 for a non-pick contact chunk
  driven by ``prompt`` (e.g. turning a knob, toggling a stove, a short
  push).
- ``move_to(xyz)`` — scripted Cartesian motion to an absolute
  world-frame ``[x, y, z]`` target in meters (deterministic; no VLA).
- ``move_pose(xyz)`` — scripted Cartesian motion that co-varies
  position and wrist orientation (pitch + yaw) at once; threads
  cabinet-front / low-shelf poses where a decoupled servo stalls.
- ``rotate_wrist(target_yaw / delta_yaw)`` — scripted wrist rotation
  around the world Z-axis; pass an absolute ``target_yaw`` or a
  relative ``delta_yaw`` (radians).
- ``rotate_pitch(target_pitch / delta_pitch)`` — scripted gripper tilt
  around the world X-axis; pass an absolute ``target_pitch`` or a
  relative ``delta_pitch`` (radians).
- ``release()`` — open the gripper.
- ``set_gripper(gripper, steps)`` — hold the current pose and drive the
  gripper command for ``steps`` env steps (e.g. to firm up a grip).
- ``back_project(row, col)`` — turn an image pixel (``row`` 0 = top,
  ``col`` 0 = left) into a 3D point in world coordinates.
- ``segment(prompt)`` — optional segmentation helper that localizes an
  object in an existing image artifact (falls back to manual
  localization when no service is configured).
- ``view_driver_state()`` — force a fresh state dump (images, depths,
  camera meta, ``states.json``).
- ``view_camera_meta(camera)`` — read camera calibration metadata
  (``agentview`` or ``wrist``) for localization.
- ``finish(status, summary)`` — end the episode; ``status`` is
  ``success`` / ``failure`` / ``stuck`` and ``summary`` is a short
  natural-language recap (both required).

Every tool re-renders the world after it runs, so the next turn's
context reflects the post-action state.

Live dashboard
--------------

Add ``--dashboard`` to open a local monitor for a LIBERO run:

.. code-block:: bash

   rpent --env libero --dashboard \
     --suite libero_goal_task --task 1 --seed 0 --planner claude_code

The dashboard streams reasoning, agentview + wrist camera + Pi0.5
overlays, and an action timeline. Use
``--dashboard-language zh-cn`` for the Chinese UI.

Bringing your own VLA
---------------------

If you have a LIBERO-compatible VLA that is not Pi0.5, swap the model
client without touching the env by:

1. Writing a new ``vla_server.py`` that exposes the same ``predict``
   RPC contract (over http or socket).
2. Pointing at it with ``--vla-endpoint [protocol://]host:port``.
3. Optionally updating ``robots/libero/toolkit.py`` if the tool
   surface (e.g. ``pi0_pick`` → ``mymodel_pick``) needs to change.

See :doc:`../development/add_primitive` for the full walkthrough.
