Overview
========

**RPent (Recursive Physical Agent)** is an open framework for building embodied 
agents that continuously evolve through recursive interaction with the physical world. 
Rather than prescribing a single foundation model, RPent provides a recursive agent 
framework that harnesses heterogeneous intelligence, including perception, reasoning, 
memory, execution, and self-evolution, into a unified physical agent. 
Through continuous interaction, reflection, and adaptation, RPent enables physical agents 
to acquire new capabilities and evolve beyond their initial design.

The name Pent is inspired by the Pentagram, whose five points symbolize the integration 
of multimodal intelligence into a unified embodied agent. At its center, the infinity 
symbol (∞) represents the endless recursive cycle of perception, reasoning, 
execution, and self-evolution, through which intelligence continuously expands into the physical world.

.. image:: https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png
   :alt: RPent framework diagram
   :align: center
   :width: 90%

RPent is built upon three core design principles: **service-oriented, standardized, and composable**. 
RPent enables capabilities to be deployed as reusable services, 
connected through unified interfaces, and flexibly composed into diverse physical agents. 
Together, these principles allow RPent to move beyond traditional robot control frameworks 
and establish an agentic infrastructure for the physical world, where intelligence 
is not only deployed, but continuously built, expanded, and evolved.

Concretely, RPent stands out in three ways:

- **The VLA is treated as a retryable tool, not an end-to-end executor.**
  The VLA weights stay frozen throughout; the policy is wrapped as action
  primitives like ``pi0_pick`` and ``pi0_doubled``, sitting alongside
  scripted tools such as ``move_to``, ``rotate_wrist``, and ``back_project``
  in one common tool schema for the planner to call on demand. Each
  environment also carries a memory of which prompts and conditions make the
  VLA reliable, so the planner knows when and how to invoke it. See
  :doc:`development/memory` for how the memory is organized and loaded.
- **One tool surface, three planners, directly comparable.**
  ``--planner {api, claude_code, codex}`` drives the exact same toolkit with
  a pydantic-ai loop, the Claude Agent SDK, or the Codex SDK. Because the
  tool surface is identical, different planners can be compared head-to-head
  on the same physical benchmark.
- **The simulation environment is isolated — swappable, portable from sim to
  real.** The simulator (or real robot) runs in a standalone env_server, kept
  separate from reasoning and model weights; swapping environments only means
  implementing the same interface, and an env can be restarted on its own,
  moved to another machine, or switched from simulation straight to hardware.
  That isolation is what lets a prototype scale up.

Feature Matrix
--------------

.. list-table::
   :header-rows: 1
   :widths: 26 28 26 20

   * - Agentic Planner
     - Action Primitive
     - Simulator
     - Real World
   * - - Claude Code ✅
       - Codex ✅
       - Custom planner ✅
     - - **VLA manipulation**

         - Pi0.5 ✅
         - RLDX-1

       - **WAM manipulation**

         - DreamZero
     - - LIBERO-PRO ✅
       - RoboCasa
     - - Franka
       - SO-101

Where to go next
----------------

- New to RPent? Start with :doc:`installation` and then
  :doc:`quickstart` to get a LIBERO task running end-to-end.
- Want to drive a specific robot or change a planner? Head to
  :doc:`usage/configure_planner`.
- Extending RPent for your own scenarios? See
  :doc:`development/architecture`.
