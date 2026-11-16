LIBERO RL Rollouts
==================

``libero_rl`` runs LIBERO-PRO with a language-model policy and six robot tools.
It uses no action model, task guides or task memory. The same robot interface
is used by standalone RPent collection and RLinf training. This page covers
collection; the RLinf repository owns the training configuration and launcher.

Overview
--------

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Action Models

      No VLA; tools control the end effector directly.

   .. grid-item-card:: Planners

      Qwen with image inputs and tool calls; the example uses Qwen3.8-27B.

   .. grid-item-card:: Tasks

      Four LIBERO-PRO suites with 40 tasks.

   .. grid-item-card:: Hardware

      Linux NVIDIA GPUs; the example assigns two GPUs to serving, one to SAM3 and one to simulation.

Tasks
~~~~~

.. list-table::
   :header-rows: 1

   * - Scope
     - Contents
   * - LIBERO-PRO
     - ``libero_object``, ``libero_goal``, ``libero_spatial`` and ``libero_10``, each with task 0–9.
   * - Seed
     - ``--seed`` selects the task initial state; batch collection can cover multiple seeds.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Item
     - Behavior
   * - Observation
     - Adaptive only: measured state at initialization and after actions; ``view_env`` returns two camera images.
   * - Action
     - ``move_to`` / ``set_gripper`` / ``view_env`` / ``segment`` / ``back_project`` / ``finish``
   * - Reward / success
     - The last state's LIBERO ``terminated`` flag determines success, not the planner's ``finish`` response.
   * - Task prompt
     - Task description and tool prompt; ``simple`` by default, with brief-reasoning and geometry variants.

Each episode starts a fresh simulator and ends on ``finish``, an exhausted budget or a timeout. Episodes can share SAM3. The example GPU allocation is not a minimum-memory guarantee; verify one task before increasing concurrency.

Installation and Resources
--------------------------

Complete :doc:`libero_rl_setup` using the paired ``Sonorous281/RPent`` and ``Sonorous281/RLinf`` integration branches. Prepare separate simulator and model-serving environments, the full Qwen model, SAM3 weights and LIBERO-PRO assets. No Pi0.5 checkpoint is needed.

Run the following commands from the RPent checkout root. Use the same source and assets in all three terminals. For custom storage, export ``RPENT_DATA_ROOT`` in each terminal before sourcing ``env.sh``.

Run a Task
----------

In a new terminal, activate SGLang and start a model service. ``serve.sh``
uses that interpreter (or ``RPENT_SGLANG_PYTHON``) and requires ``MODEL_PATH``.
Its defaults are the name Qwen3.8-27B, GPU 2–3, TP2, MTP and port 30080.
``SERVE_GPUS``, ``SERVE_PORT``, ``MODEL_PATH``, ``MODEL_NAME``,
``SERVE_MEM_FRACTION`` and ``SERVE_CONCURRENCY`` override the service settings.
For a model without MTP weights, use ``ENABLE_MTP=0``.

.. code-block:: bash

   source .venv-sglang/bin/activate
   MODEL_PATH="$HOME/.local/share/rpent/models/Qwen3.8-27B" SERVE_GPUS=2,3 \
     bash tools/libero_rl_eval/serve.sh

Set ``MODEL_PATH`` to the downloaded directory when using ``XDG_DATA_HOME`` or custom storage. Keep this terminal in the SGLang environment; ``env.sh`` requires the RPent interpreter and is only sourced in the simulator/SAM3 terminals.

The service is ready when ``curl http://127.0.0.1:30080/v1/models`` lists the
model. In a second terminal, activate RPent and start shared SAM3:

.. code-block:: bash

   source .venv/bin/activate
   source tools/libero_rl_eval/env.sh
   python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8112

In a third terminal, activate RPent and run one episode on GPU 1:

.. code-block:: bash

   source .venv/bin/activate
   source tools/libero_rl_eval/env.sh
   export RPENT_MODEL=openai-chat:qwen3.8-27b
   export RPENT_BASE_URL=http://127.0.0.1:30080/v1
   export RPENT_SAM3_ENDPOINT=http://127.0.0.1:8112
   bash tools/libero_rl_eval/run-libero-rl.sh \
     --suite libero_object --task 0 --seed 0 --cuda-device 1

``--cuda-device`` selects the simulator GPU. The SAM3 endpoint selects the
shared service independently. Omitting it starts an episode-owned SAM3 server
on the simulator GPU. Extra command-line arguments pass through to ``rpent``.

View Results
------------

The default output is ``logs/libero-rl-adaptive-<prompt>-<time>`` in the RPent
checkout. Set ``RPENT_OUTPUT_DIR`` to choose another directory. Each episode
contains:

* ``result.json``: native success, errors, turns and token usage;
* ``states.json``: measured states and native ``terminated`` flags;
* ``transcript_*.json``: native messages with thinking and tool calls;
* ``rollout.json``, ``episode.mp4`` and camera/state artifacts;
* ``trajectories/<task_id>/``: the task outcome, planner requests and responses,
  tool calls, executed actions, observations and their attachments.

The launcher enables RPent's shared trajectory recorder. Planner records contain
Pydantic AI inputs after history compaction and the corresponding responses;
these are SDK records, not raw HTTP payloads. Task metadata and linked records use schema version 5.
The audit requires one completed task, a usable response for every planner turn,
readable attachments and agreement with native success before the batch launcher
publishes the episode. Historical ``api_trace/`` episodes remain readable with
their original wire-response and checksum checks.

Collect Tasks and Seeds
-----------------------

``jobs/panel40.txt`` lists task 0–9 from ``libero_object``, ``libero_goal``,
``libero_spatial`` and ``libero_10`` in adaptive mode at seed 0. Each manifest
line has ``adaptive suite task seed``. ``--seeds`` overrides the listed
seed with a comma-separated list or inclusive ranges.

.. code-block:: bash

   python tools/libero_rl_eval/run_batch.py run \
     "$RPENT_DATA_ROOT/rollouts/new-run" \
     --jobs tools/libero_rl_eval/jobs/panel40.txt \
     --seeds 3-66 --gpus 1 --per-gpu 4

The model and shared SAM3 variables from the single-episode example also apply
here. ``--endpoints URL,URL`` distributes workers over additional local model
services. ``--start-interval`` staggers initial worker starts; size
``--per-gpu`` for the simulator GPU's available memory and initialization load.

.. list-table::
   :header-rows: 1

   * - Batch argument
     - Default
     - Meaning
   * - ``--jobs``
     - Required
     - Four-field task manifest
   * - ``--seeds``
     - Manifest seeds
     - Comma-separated seeds/ranges, e.g. ``0,3-66``
   * - ``--gpus``, ``--per-gpu``
     - GPUs required; 1 worker/GPU
     - Environment GPUs and worker count per GPU
   * - ``--prompt``
     - ``simple``
     - One prompt variant for the batch
   * - ``--endpoints``
     - ``RPENT_BASE_URL``
     - Comma-separated policy ``/v1`` endpoints
   * - ``--start-interval``
     - ``1`` second
     - Spacing between initial worker starts
   * - ``--episode-timeout``
     - ``14400`` seconds
     - Outer process timeout per episode
   * - ``--infrastructure-retries``
     - ``1``
     - Additional attempts after an error/audit failure per invocation

Each job has separate ``attempt-NNN`` directories. A normal unsuccessful task
is valid data. An infrastructure error or failed audit is retained with
``INVALID.json`` and retried once by default. An interrupted attempt gets
``INTERRUPTED.json``. Only audited episodes receive ``COMPLETED.json`` and an
entry in the atomically updated ``valid-index.json``. Completed data is usable
when the batch is paused. SIGINT/SIGTERM terminate owned episode process groups;
repeat the same command to resume. A directory lock prevents two collectors
from launching the same queue. The manifest rejects changed task, model,
prompt or generation settings on resume.

.. code-block:: bash

   python tools/libero_rl_eval/run_batch.py summary \
     "$RPENT_DATA_ROOT/rollouts/new-run"

The summary re-audits sealed episodes and reports valid episodes and native
successes. Use the paired ms-swift checkout's ``ms-swift/examples/rpent`` for SFT data preparation and training on the collected trajectories.

Model Service Configuration
---------------------------

The model service listens on ``127.0.0.1``. These variables control serving; arguments after ``serve.sh`` pass through to SGLang.

.. list-table::
   :header-rows: 1

   * - Service variable
     - Default
     - Meaning
   * - ``RPENT_SGLANG_PYTHON``
     - ``python``
     - Interpreter from the active SGLang environment; does not activate CUDA libraries
   * - ``SERVE_GPUS``
     - ``2,3``
     - Visible physical GPUs; their count sets TP
   * - ``MODEL_PATH``
     - Required
     - Local checkpoint directory
   * - ``MODEL_NAME``
     - ``qwen3.8-27b``
     - Served name; match ``RPENT_MODEL`` after ``openai-chat:``
   * - ``SERVE_PORT``
     - ``30080``
     - Local model port
   * - ``SERVE_MEM_FRACTION``
     - ``0.8``
     - SGLang static memory fraction
   * - ``SERVE_CONCURRENCY``
     - ``8``
     - Maximum running model requests
   * - ``ENABLE_MTP``
     - ``1``
     - EAGLE/MTP: 3 steps, top-k 1, 4 draft tokens; set 0 for a model without MTP weights

Planner Configuration
---------------------

``planner.sh`` supplies Qwen thinking sampling: temperature 1.0, top_p 0.95,
top_k 20. Qwen3.8 uses ``reasoning_effort=xhigh`` by default; other Qwen models
send no effort tier, while thinking remains enabled. The service uses
``qwen3_coder`` for tool calls and ``qwen3`` for reasoning.

.. list-table:: Overrides
   :header-rows: 1
   :widths: 45 55

   * - Variable
     - Default
   * - ``RPENT_PROMPT``
     - ``simple``
   * - ``RPENT_MAX_TURNS`` / ``RPENT_MAX_EPISODE_STEPS``
     - 100 / 10000
   * - ``RPENT_MAX_TOKENS`` / ``RPENT_CONTEXT_WINDOW``
     - 32768 / 262144
   * - ``RPENT_COMPACTION_FRACTION``
     - 0.7
   * - ``RPENT_REASONING_EFFORT``
     - ``xhigh`` for Qwen3.8; ``none`` for other models
   * - ``RPENT_PLANNER_TIMEOUT_S``
     - 7200
   * - ``RPENT_PRESENCE_PENALTY`` / ``RPENT_REPETITION_PENALTY``
     - Unset; optional explicit sampling overrides

To reproduce the SFT-checkpoint evaluation penalties, set ``RPENT_PRESENCE_PENALTY=1.5`` and ``RPENT_REPETITION_PENALTY=1.0`` explicitly. ``--observation-mode adaptive`` is accepted; image/text modes are unsupported. Reduce ``RPENT_MAX_TURNS`` for an installation check, while keeping the intended budget for collection.

Earlier thinking is preserved in ``reasoning_content`` and rendered by the
model's own chat template. RPent wraps tool-image user messages with
``<tool_response>`` tags so Qwen3.5 retains earlier thinking. Qwen3.8 also uses
``preserve_thinking=true``. Only context management intentionally changes the
visible history; the resulting SDK request is retained in the trajectory.
A reply ending at the output-token limit remains an incomplete training
output and fails the trajectory audit.

The prompt definitions live in ``robots/libero_rl/prompt_bundle.py``.
``simple`` contains the tool contract; ``simple_brief`` also asks for at most
256 reasoning tokens per decision. This is prompt guidance, not a decoder
limit. ``base`` adds robot frames and an
execution method; ``base_gripper`` adds finger geometry; and
``base_gripper_camera`` also adds camera geometry.

For RLinf message-window recipes, the compaction trigger includes estimated
image tokens, rather than treating image bytes as text. The message cutoff
still follows Harness's sliding window and preserves the initial task. This
estimate is used for context management; it is not measured model usage.
Token-window recipes retain Harness's token-window behavior.

RLinf Training
--------------

Use the paired branch's `RLinf multimodal training guide <https://github.com/Sonorous281/RLinf/blob/dev/rpent-rlinf-integration/docs/source-en/rst_source/examples/agentic/rpent_libero_rl.rst>`__ for installation, launch, recovery and evaluation. RPent owns simulation and tools; RLinf owns generation, actor/critic updates and weight synchronization. Set ``tools.rpent.python`` to the RPent interpreter and provide the subprocess CUDA/EGL environment.

Standalone collection defaults to LIBERO-PRO; the RLinf multimodal recipe defaults to standard LIBERO. Collection environment variables do not override RLinf settings. Before comparing results, match the benchmark, tasks and seeds, model, ``agentloop.prompt``, ``agentloop.chat_template_kwargs``, ``agentloop.history_compaction``, ``algorithm.sampling_params`` and ``runner.seq_length``.

For SFT, use ``ms-swift/examples/rpent/README.md`` in the paired ms-swift checkout. Exporters must support schema version 5 and preserve post-compaction requests, image content/order, historical thinking and assistant loss masks.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1

   * - Symptom
     - Check
   * - Model endpoint unavailable
     - Check ``/v1/models`` for the configured model name, then inspect the serving terminal and its separate environment.
   * - Simulator initialization fails
     - Check EGL/driver availability and the asset, BDDL and initial-state paths in ``config.yaml``.
   * - Attempt marked invalid
     - Read ``INVALID.json`` and the attempt log. A normal task failure is valid; truncated responses, missing attachments and infrastructure errors need repair before retrying.

.. toctree::
   :hidden:

   libero_rl_setup
