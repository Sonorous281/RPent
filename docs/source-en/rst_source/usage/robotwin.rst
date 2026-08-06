RoboTwin
========

RPent uses RLinf's ``RoboTwinEnv`` as the sole owner of the RoboTwin native
task. The RPent process only talks to the environment through a thin RPC
bridge:

.. code-block:: text

   RPent Toolkit -> RPent EnvServer -> RLinf RoboTwinEnv
   -> RoboTwin VectorEnv/SubEnv -> native task

Requirements
------------

Use the three matching ``adapt/robotwin-hybrid`` branches of RPent, RLinf, and
RoboTwin. The RoboTwin branch is based on
``RLinf_support@0008ae6800df9f75fc8de7098bacb01735fd8fd2`` and carries the
compatibility patch described by
``compatibility/rpent_downloads_manifest.json``. Do not point a parity run at
an arbitrary RoboTwin checkout.

Download the pinned LingBot checkpoint:

.. code-block:: bash

   hf download RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500 \
      --revision c55199f25a10397e79dce177ee11c8774fb8edde \
      --local-dir /path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500

The LingBot source checkout supplied with the checkpoint must expose
``deploy.websocket_client_policy`` and ``deploy.lingbot_vla_policy``. RPent
uses the official websocket client and server; normalization stays inside
that runtime.

Run
---

Set explicit source and interpreter paths, then launch one hybrid episode:

.. code-block:: bash

   export ROBOTWIN_ROOT=/path/to/RoboTwin
   export ROBOTWIN_ASSETS_ROOT=/path/to/pinned/RoboTwin-assets
   export ROBOTWIN_CUROBO_ROOT=/path/to/curobo
   export RPENT_RLINF_ROOT=/path/to/RLinf
   export VLA_ROOT=/path/to/LingBot-VLA-source
   export LINGBOT_MODEL_PATH=/path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500

   rpent --env robotwin \
      --task-name place_fan \
      --task-config demo_randomized \
      --seed 100002 \
      --seed-mode exact \
      --allow-infeasible \
      --instruction "$ROBOTWIN_INSTRUCTION" \
      --robotwin-hybrid-workspace "$ROBOTWIN_ROOT/hybrid_workspace" \
      --env-python /path/to/robotwin/python \
      --vla-python /path/to/lingbot/python \
      --env-cuda-device 0 \
      --vla-cuda-device 2 \
      --planner codex \
      --model gpt-5.5 \
      --reasoning-effort high

Formal Downloads parity uses ``demo_randomized``, exact native seeds,
``--allow-infeasible``, and the instruction recorded in the frozen seed table.
Omitting ``--instruction`` is supported for exploratory runs and generates the
native instruction deterministically from the selected seed.

``--allow-reset`` enables the controlled same-seed reset tool. It is disabled
by default. ``--env-endpoint`` and ``--vla-endpoint`` attach to existing
services, but effect-parity runs should launch the pinned local services so
their source and model identity are auditable.
``--cuda-device`` keeps the legacy same-GPU behavior. Do not combine it with
``--env-cuda-device`` or ``--vla-cuda-device``. Formal paired runs record both
physical GPU indices and UUIDs in ``robotwin_resource_manifest.json``.

``ROBOTWIN_ROOT`` selects the compatibility source checkout.
``ROBOTWIN_ASSETS_ROOT`` selects the separately pinned root containing
``assets/``; it may equal ``ROBOTWIN_ROOT`` for a complete packaged checkout.
``ROBOTWIN_CUROBO_ROOT`` must be a clean
``NVlabs/curobo@2fbffc35225398cf9d5f382804faa9de2608753b`` checkout. Hybrid
startup imports its ``src`` tree and rejects any other revision. Planning also
requires ``warp-lang==1.11.1`` for that pinned cuRobo revision.
``--robotwin-hybrid-workspace`` is read only by the RPent runtime. The runtime
imports only allowlisted semantic-recipe fields after rejecting legacy
protocol markers, paths, and endpoints. The source ``GUIDE.md``, memory index,
saved coordinates, and simulator oracle state never enter the Agent prompt.

Runtime Contract
----------------

The startup handshake requires ``robotwin-agent-v1``,
``downloads_hybrid``, and compatibility ID
``robotwin-rpent-downloads-2026-07-31-v1``. The native action layouts are
``qpos14`` and world-frame ``eef16`` with ``wxyz`` quaternions.

Each state-changing call has a server-scoped mutation ID. After a transport
timeout, RPent queries that ID instead of replaying the action. If native
state becomes unknown, camera and robot-state access stop; only status and a
controlled reset are allowed.

Task success is always a fresh ``TASK_ENV.eval_success`` result. Completion of
a VLA chunk or primitive does not imply task success.
