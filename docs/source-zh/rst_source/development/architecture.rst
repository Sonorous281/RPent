系统设计
========

.. raw:: html

   <div style="text-align: center;">
     <img src="https://github.com/RLinf/misc/raw/main/pic/rpent_framework.png" alt="RPent 框架"
          style="max-width: 95%; height: auto;" />
   </div>

框架概览
--------

**RPent (Recursive Physical Agent)** 是一个用于构建具身智能体的开源框架，
让智能体通过与物理世界的递归交互持续进化。它把
感知、推理、记忆、执行、自我进化这些异构能力融合到一个统一的物理智能体中。
凭借持续的交互、反思与适应，智能体能够习得新能力，超越其初始设计。

RPent 建立在三条核心设计原则之上。

**服务化。** 各项能力都以可复用的服务形式部署，比如 VLA 策略服务、环境
服务，每一个都能独立扩缩容、重启或替换。

**标准化。** 各项服务通过统一的接口连接：一套 ``Planner`` 协议、一套通用的
``Toolkit`` API、一套共享的 RPC 底座，异构能力因此无需胶水代码就能拼到一起。

**可组合。** 这些服务能拼装成各式各样的物理智能体：换一个 planner、混用来自
不同 VLA 的动作 primitive、接入新仿真器、加上记忆，都不必改动其余部分。

这三条原则让 RPent 跳出传统机器人控制框架的范畴，成为一套面向物理世界的
智能体基础设施：智能不只是被部署，还在被持续地构建、扩展和进化。

下面几节讲清楚这套架构怎么落地：运行时各部分如何分工、彼此如何通信，以及
代码在 ``rpent/`` 和 ``robots/`` 下如何组织。

关键特性
--------

这一节聚焦 RPent 区别于其他具身智能体框架的几个设计选择。

**LLM 作为 planner。** 这是 RPent 与多数具身智能体最根本的不同：后者端到端训练
一个策略模型直接输出动作，RPent 则让一个通用 LLM 充当 planner，靠推理
和工具调用来指挥机器人，VLA、脚本化动作只是它能调用的底层能力。每次工具调用的
返回（文本加图像）都喂回给它，让它对着实际看到的画面决定下一步。这样就用上了
LLM 的通用推理和临场纠错，而不必为每个新任务重新训练模型。

planner 本身也可以替换，内置三种后端：

- **自研 agent loop**：RPent 自己的工具调用循环，与具体 provider 无关。
- **Claude Agent SDK**：复用 Anthropic 官方的 agent runtime。
- **Codex SDK**：复用 OpenAI Codex 的 agent runtime。

三者的取舍和具体配置，见 :doc:`../usage/configure_planner`。

**环境解耦。** 仿真器和真机都作为独立的 env_server 运行，只通过一套轻量 RPC
和 agent 通信；agent 这边不 import 任何仿真器，也不绑定具体环境。于是换环境只要
实现同一套 ``EnvClient`` 接口，env 就能单独重启、迁到另一台机器，或从仿真直接切到
真机，planner 和工具都不用动。GPU 上的策略（vla_server）同样是独立进程，和
推理、仿真互不拖累。新增环境甚至不用改注册代码，把包放进 ``robots/`` 目录框架就会
自动发现，详见 :doc:`add_robot`。

**实时监控。** 加上 ``--dashboard`` 会启动一个本地 FastAPI 监控页，实时展示 LLM
的推理过程、相机与 Pi0 视图，以及动作时间线，界面支持中英双语
（``--dashboard-language {en, zh-cn}``）。

智能体循环
----------

一次运行就是一段 LLM 全程在环的循环：

1. LLM 分析任务，调用一个工具，比如 ``pi0_pick``。
2. 工具背后的 primitive driver 向 vla_server 请求一段动作 chunk（``predict``）。
3. env_server 执行这段 chunk（LIBERO 的 ``chunk_step`` 一次走完整段）。
4. 环境渲染出新的观测和相机画面。
5. 结果组装成文本加图像的内容块，喂回给 LLM 进入下一轮。

循环有两种收尾方式：LLM 调用 ``finish`` 主动结束，把状态标为 ``success``、
``failure`` 或 ``stuck``；或者跑满 ``--max-turns``、``--max-episode-steps``
设定的上限。

仓库布局
--------

代码按关注点拆分得比较清爽：

.. code-block:: text

   rpent/
     planner/       # 决策大脑：api_loop、claude_code、codex、base。
     cli/            # main.py 入口 (无 __init__.py，不是 subpackage)。
     context/        # Prompt bundle、prompt 工具、共享 prompt 分节。
     dashboard/      # FastAPI 监控加 SSE 流 (可选)。
     envs/           # EnvSpec、PromptBundle 与惰性 env 注册表。
     tools/          # Toolkit 基类和共享的工具辅助函数。
     utils/          # 配置、日志、RPC 客户端与服务端、VLA HTTP shim。
   robots/
     libero/         # LIBERO 的一整套实现，参考环境。
     (robocasa/)     # RoboCasa 驱动 —— 研发中。
     (franka/)       # Franka 驱动 —— 研发中。
     (so101/)        # SO-101 驱动 —— 研发中。
   scripts/          # 安装脚本 (LIBERO PRO/PLUS、codex proxy)。

Runner
------

``rpent/cli/main.py`` 是整场运行的编排者，负责把三个进程拉起来、接上线，
再交给推理循环。它先拉起 env_server 和 vla_server 两个子进程，等它们
就绪后，为所选环境构造 toolkit，并按参数选定的 planner 构造决策大脑，最后跑起
工具调用循环，把结果落盘。日常会用到的命令行参数在 :doc:`../quickstart` 里介绍。

Runner 有意保持轻薄：跟环境有关的东西都在 ``robots/<env>/`` 下，跟大脑有关的
都在 ``rpent/planner/`` 下。

Dashboard（可选）
-----------------

``rpent/dashboard/`` 是一个 FastAPI 应用加一份静态前端。开了 ``--dashboard``
时，``rpent/cli/main.py`` 会把它绑定到指定的主机和端口（默认本地、随机端口），
先弹出一个选配置的启动页，随后开始推送：

- 智能体的推理 token（走 SSE）。
- 实时的相机与 Pi0.5 视图。
- 动作时间线，点击其中任一动作可回看那一步的画面。

.. note::

   Dashboard 是纯观察性的，永远不影响循环，所以即便它内部出错，也不会拖垮
   一次运行。

下一步
------

- 想了解扩展 RPent 要对接的接口？见 :doc:`interfaces`。
- 想新增机器人？见 :doc:`add_robot`。
- 想新增一个 VLA 或动作 primitive？见 :doc:`add_primitive`。
- 想了解记忆的设计和接入点？见 :doc:`memory`。
- 想要完整的扩展清单？见 :doc:`add_robot`。
