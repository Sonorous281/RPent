概览
====

**RPent (Recursive Physical Agent)** 是一个用于构建具身智能体的开源框架，
让智能体通过与物理世界的递归交互持续演进。它不预设某个具体的基础模型，而是
把感知、推理、记忆、执行、自我演进这些异构能力统一到一个物理智能体中。凭借
持续的交互、反思与适应，智能体能够获得超出其初始设计的新能力。

Pent 这个名字取自五芒星（Pentagram），五个顶点象征多模态智能融合成一个统一的
具身智能体。五芒星中心是无穷符号（∞），代表感知、推理、执行、自我演进这个
永不停歇的递归循环，让智能持续向物理世界扩展。

.. image:: https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png
   :alt: RPent 框架图
   :align: center
   :width: 90%

RPent 建立在三条核心设计原则之上：服务化、标准化、可组合。它把各种能力以可复用
服务的形式部署，通过统一接口连接，再灵活组合成多样的物理智能体。正是这三条原则
让 RPent 跳出传统机器人控制框架的范畴，成为一套面向物理世界的智能体基础设施：
智能不只是被部署，还在被持续地构建、扩展与演进。

具体来说，RPent 有三点与众不同：

- **VLA 被当作可重试的工具，而非端到端执行器。** VLA 权重全程冻结，封装成
  ``pi0_pick``、``pi0_doubled`` 这样的动作 primitive，和 ``move_to``、
  ``rotate_wrist``、``back_project`` 等脚本化工具并列在同一套 tool schema 里，
  由 planner 按需调用。每个环境还配一份记忆，记录什么样的 prompt、在什么条件下能让 VLA
  稳定工作，planner 据此知道何时、如何调用它。记忆怎么组织、怎么加载，见
  :doc:`development/memory`。
- **同一套工具，三种 planner，可直接横比。** ``--planner {api, claude_code,
  codex}`` 让完全相同的 toolkit 分别跑在 pydantic-ai 循环、Claude Agent SDK、
  Codex SDK 之上。因为工具完全相同，不同 planner 可以在同一个物理 benchmark
  上直接对比。
- **仿真环境是隔离的，能独立替换、从仿真走到真机。** 仿真器（或真机）跑在
  独立的 env_server 里，和推理、模型权重都隔开；换环境只要实现同一套接口，它
  就能单独重启、迁到别的机器，或从仿真直接切到真机。正是这种隔离，让 RPent 能
  从原型走向规模化。

功能矩阵
--------

.. list-table::
   :header-rows: 1
   :widths: 26 28 26 20

   * - Agentic Planner
     - Action Primitive
     - Simulator
     - 真实机器人
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

下一步
------

- 第一次接触 RPent，先看 :doc:`installation`，再跟着 :doc:`quickstart`
  端到端跑通一个 LIBERO 任务。
- 要驱动某个具体机器人或切换 planner，直接看 :doc:`usage/configure_planner`。
- 打算基于 RPent 做扩展，看 :doc:`development/architecture`。
