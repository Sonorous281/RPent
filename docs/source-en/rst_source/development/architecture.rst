System Internals
================

.. raw:: html

   <div style="text-align: center;">
     <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png" alt="RPent framework"
          style="max-width: 95%; height: auto;" />
   </div>

Framework overview
------------------

**RPent (Recursive Physical Agent)** is an open framework for building
embodied agents that continuously evolve through recursive interaction
with the physical world. Rather than prescribing a single foundation
model, RPent provides a recursive agent framework that harnesses
heterogeneous intelligence — perception, reasoning, memory, execution,
and self-evolution — into a unified physical agent. Through continuous
interaction, reflection, and adaptation, RPent enables physical agents
to acquire new capabilities and evolve beyond their initial design.

RPent is built upon three core design principles:

- **Service-oriented.** Capabilities are deployed as reusable services
  (e.g. the VLA policy server, the environment server, the memory
  corpus) that can be independently scaled, restarted, or replaced.
- **Standardized.** Every service connects through unified interfaces
  — a single ``Planner`` protocol, a common ``Toolkit`` API, and a
  shared RPC substrate — so heterogeneous intelligence modules compose
  without glue code.
- **Composable.** Services are flexibly assembled into diverse physical
  agents: swap the planner, mix primitives from different VLAs, plug in
  a new simulator, or add memory — all without touching the rest of the
  stack.

Together, these principles allow RPent to move beyond traditional robot
control frameworks and establish an *agentic infrastructure for the
physical world*, where intelligence is not only deployed, but
continuously built, expanded, and evolved.

The sections below walk through how this architecture is implemented:
what each part does at runtime, how they communicate, and how the code
is organized under ``rpent/`` and ``robots/``.

Key features
------------

This section highlights the design choices that set RPent apart from
other embodied-agent frameworks.

**VLA as a retryable tool.** Instead of training an end-to-end policy model
that outputs actions directly, RPent has a general LLM act as the planner and
call the frozen VLA as action primitives like ``pi0_pick`` and ``pi0_doubled``,
alongside scripted tools such as ``move_to`` and ``rotate_wrist`` in one common
tool schema. Each call returns text plus images, fed back so the LLM decides
its next step from what it actually sees; together with a per-environment
memory, the planner knows under what conditions the VLA is reliable. This taps
the LLM's general reasoning and on-the-fly recovery without retraining a model
per task. See :doc:`add_primitive` for how to add a new primitive.

**Swappable planner, three symmetric backends.** Changing one ``--planner``
argument switches the agent runtime behind the scenes, leaving the tools and
prompts untouched. Three are built in: a built-in agent loop (RPent's own
tool-calling loop), the Claude Agent SDK, and the Codex SDK — the latter two
reuse the official runtimes. Because all three face the exact same tools, they
can be compared head-to-head on the same physical benchmark. See
:doc:`../usage/configure_planner` for the trade-offs and configuration.

**Environment decoupling, sim to real.** The simulator or real robot runs as a
standalone ``env_server`` that talks to the agent over lightweight RPC; the
agent side imports no simulator and isn't tied to any specific environment.
Swapping environments just means implementing the same ``EnvClient`` interface
— an env can be restarted on its own, moved to another machine, or switched
from simulation straight to hardware, without touching the planner or tools.
Adding an environment doesn't even need registration code: drop a package under
``robots/`` and the framework discovers it. See :doc:`add_robot` for how to
wire up a new environment.

The agentic loop
----------------

A single run is an LLM-in-the-loop cycle:

1. The LLM reasons about the task and calls a tool
   (e.g. ``pi0_pick``).
2. The tool's **primitive driver** asks the ``vla_server`` for an
   action chunk (``predict``).
3. The ``env_server`` executes that chunk (LIBERO's ``chunk_step`` runs
   the whole chunk in one shot).
4. The env renders the resulting observation and camera frames.
5. Results are turned into text + image content blocks and fed back
   to the LLM for the next turn.

The loop ends when the LLM calls the ``finish`` tool
(``success`` / ``failure`` / ``stuck``) or hits ``--max-turns`` /
``--max-episode-steps``.

Repository layout
-----------------

The code that implements the framework is split cleanly by concern:

.. code-block:: text

   rpent/
     planner/       # Reasoning brains: api_loop, claude_code, codex, base.
     cli/            # main.py entrypoint (no __init__.py — not a subpackage).
     context/        # Prompt bundles, prompt utils, shared prompt sections.
     dashboard/      # FastAPI monitor + SSE streams (optional).
     envs/           # EnvSpec, PromptBundle, and the lazy env registry.
     tools/          # Toolkit base class and shared tool helpers.
     utils/          # Config, logging, RPC client/server, VLA HTTP shim.
   robots/
     libero/         # LIBERO env_client / env_server / vla_server /
                     # toolkit / prompt_bundle. The reference env.
     (robocasa/)     # RoboCasa driver — in progress.
     (franka/)       # Franka driver — in progress.
     (so101/)        # SO-101 driver — in progress.
   scripts/          # Setup scripts (LIBERO PRO/PLUS, codex proxy).

The runner
----------

``rpent/cli/main.py`` is the choreographer: it brings the three
processes up, wires them together, and hands off to the reasoning loop.
It first spawns the ``env_server`` and ``vla_server`` subprocesses and
waits for them to be ready, then builds the toolkit for the chosen env,
constructs the planner selected by ``--planner``, and runs the
tool-calling loop, writing the transcript and ``episode.mp4`` on exit.
The CLI flags you'll use day-to-day are documented in
:doc:`../quickstart`.

The runner is intentionally thin: everything env-specific lives under
``robots/<env>/``, and everything brain-specific lives under
``rpent/planner/``.

Dashboard (optional)
--------------------

``rpent/dashboard/`` is a FastAPI app plus a static frontend. When
``--dashboard`` is set, ``rpent/cli/main.py`` binds it on
``--dashboard-host:--dashboard-port`` (default localhost, random
port), boots a launcher page for picking config, and then streams:

- The agent's reasoning tokens (SSE).
- Live camera / Pi0.5 views.
- An action timeline; click any action to replay that step.

.. note::

   The dashboard is purely observational and never affects the loop, so
   a failure inside the dashboard cannot break a run.

From here
---------

- The interfaces you implement to extend RPent? — :doc:`interfaces`.
- Adding a new robot? — :doc:`add_robot`.
- Adding a new VLA / action primitive? — :doc:`add_primitive`.
- Curious how memory is designed and where to hook it? —
  :doc:`memory`.
- Need the full-detail extension checklist? — :doc:`add_robot`.
