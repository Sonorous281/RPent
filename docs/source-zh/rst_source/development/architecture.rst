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
通过与物理世界的递归交互实现持续进化。RPent 不预设某一个基础模型，而是提供
一套递归 agent 框架，将异构智能 —— 感知、推理、记忆、执行和自我进化 ——
融合为一个统一的物理智能体。通过持续的交互、反思和适应，RPent 使物理智能体
能够习得新能力，超越其初始设计。

RPent 建立在三条核心设计原则之上：

- **服务化 (Service-oriented)。** 能力被部署为可复用的服务 (如 VLA 策略
  server、环境 server、memory 语料库)，可以独立扩缩容、重启或替换。
- **标准化 (Standardized)。** 所有服务通过统一的接口连接 —— 一套
  ``Planner`` 协议、一套通用 ``Toolkit`` API、一套共享 RPC 底座 ——
  使异构智能模块无需胶水代码即可组合。
- **可组合 (Composable)。** 服务可被灵活组装为多样的物理智能体：换
  planner、混用来自不同 VLA 的 primitive、接入新仿真器、或添加 memory ——
  都不需要动其余部分。

这三条原则使 RPent 超越传统机器人控制框架，建立起一种面向物理世界的
*agentic infrastructure*: 智能不仅被部署，还被持续地构建、扩展和进化。

下面各节详述这套架构如何落地：三个进程各自持有什么、如何通信，以及
``rpent/`` 与 ``robots/`` 下的代码如何组织。

关键特性
--------

*(这些是架构设计围绕的框架级承诺；下面各节则展示每一项是如何落地的。)*

- **LLM-in-the-loop 控制。** LLM 不做微调 —— 它纯粹通过调工具
  (``pi0_pick``、``move_to``、``rotate_wrist``、``back_project``、
  ``finish`` 等) 来驱动机器人。每个工具的返回都以多模态上下文
  (文本 + 渲染图) 喂回，让模型基于实际观测到的世界推理。
- **三进程架构。** **Agent 进程** (LLM planner + toolkit，无 GPU
  依赖)、**env_server** (仿真器 + EGL 渲染)、**vla_server**
  (GPU 策略权重) 是三个独立进程，用轻量 RPC 串起来。任一重量级
  进程都可以独立重启、迁到另一张 GPU、或指向远程主机。
- **可插拔的 reasoning brain (planner)。** 用一个 flag ——
  ``--planner {api, claude_code, codex}`` —— 就能换决策 brain，不用
  动 tool 或 prompt:

  - ``api`` —— 基于 `pydantic-ai <https://ai.pydantic.dev/>`_ 的
    provider-无关 tool-calling 循环 (Anthropic / OpenAI / OpenAI 兼容)，
    带 prompt 缓存和历史图片剪枝。
  - ``claude_code`` —— `Claude Agent SDK
    <https://docs.claude.com/en/api/agent-sdk/overview>`_，
    把 toolkit 暴露为 in-process MCP server。
  - ``codex`` —— OpenAI Codex SDK，通过 HTTP MCP server 桥接到
    toolkit。
- **一份契约、多个 environment。** env/vla 进程拆分是一份通用契约：
  LIBERO (Pi0.5 走 HTTP) 是已发布的参考实现；RoboCasa (RLDX-1 走
  socket-RPC) 正在开发中。添加新环境只需实现相同接口 —— 只有传输协议
  随各 env 的 observation 形状改变。
- **实时 dashboard。** 可选的 ``--dashboard`` 会起一个本地 FastAPI
  监控页，实时展示 agent 的 reasoning、相机 / Pi0 视图、动作时间线、
  剪辑回放 —— 提供 **双语 UI** (``--dashboard-language {en, zh-cn}``)。
- **加一个 environment 只需把包放进硬盘。** 没有中央注册表要改 ——
  见 :doc:`add_robot`。

单轮循环是怎么发生的
--------------------

一次运行就是一段 LLM-in-the-loop 循环：

1. LLM 分析任务、调一个工具 (如 ``pi0_pick``)。
2. 工具的 **primitive driver** 向 ``vla_server`` 请求一个 action
   chunk (``predict``)。
3. ``env_server`` 执行这段 chunk (LIBERO 是 ``chunk_step``, RoboCasa
   是逐步 ``step``)。
4. Env 渲染出新的 observation 与相机帧。
5. 结果被组装成 text + image content block，喂回 LLM 进入下一轮。

循环在 LLM 调 ``finish`` (``success`` / ``failure`` / ``stuck``)
或触达 ``--max-turns`` / ``--max-episode-steps`` 时结束。

仓库布局
--------

实现按关注点拆分得比较干净：

.. code-block:: text

   rpent/
     planner/       # 决策大脑：api_loop、claude_code、codex、base。
     cli/            # main.py 入口 (无 __init__.py，不是 subpackage)。
     context/        # Prompt bundles、prompt 工具、共享 prompt 分节。
     dashboard/      # FastAPI 监控 + SSE stream (可选)。
     envs/           # EnvSpec、PromptBundle、以及 env 的 lazy 注册表。
     tools/          # Toolkit 基类和共享 tool 辅助函数。
     utils/          # 配置、日志、RPC client/server、VLA HTTP shim。
   robots/
     libero/         # LIBERO 的 env_client / env_server / vla_server /
                     # toolkit / prompt_bundle。参考实现。
     (robocasa/)     # RoboCasa 驱动 —— 研发中。
     (franka/)       # Franka 驱动 —— 研发中。
     (so101/)        # SO-101 驱动 —— 研发中。
   scripts/          # 安装脚本 (LIBERO PRO/PLUS、codex proxy)。

Runner
------

``rpent/cli/main.py`` 是编排者。每一次调用它会：

1. 解析 CLI flag (:doc:`../quickstart` 说明了日常最常用的那些)。
2. 创建 per-run 的 scratch 目录 (``--output-dir`` 或
   ``runs/`` 下自动生成的目录)。
3. 在 loopback 上预分配一个空闲端口，把端口通过 CLI 传给子进程，
   spawn **env_server**，然后通过
   :func:`rpent.utils.rpc.wait_for_ready` 轮询 ``healthz`` 直到子进程
   起来。
4. 对 **vla_server** 做同样的事；复用已有实例时用 ``--vla-endpoint``。
5. 通过 env 的 ``get_toolkit(primitives_kwargs=...)`` 工厂为选中的
   env 构造 **toolkit**，把 env client 和 VLA client 传进去。
6. 通过 ``rpent.planner.base.build_planner`` 构造 **planner**，
   根据 ``--planner`` 选出 ``api_loop.py`` / ``claude_code.py`` /
   ``codex.py`` 之一。
7. 跑 tool-calling 循环；如果开了 ``--dashboard`` 就 stream 到 dashboard；
   结束时写出 ``<output_dir>/transcript_*.json`` 和
   ``<output_dir>/episode.mp4``。

Runner 有意保持薄：一切与 env 相关的东西在 ``robots/<env>/`` 下，
一切与 brain 相关的东西在 ``rpent/planner/`` 下。

Env 侧的注册表
--------------

``rpent/envs/base.py`` 维护一个以 env 名为 key 的 **lazy** 注册表。
传入 ``--env myenv`` 时，它会按需 import ``robots.myenv`` 包，并调用
包暴露的两个工厂：

.. code-block:: python

   # robots/myenv/__init__.py
   def get_env_spec() -> EnvSpec: ...
   def get_toolkit(*, primitives_kwargs, video_path=None, dashboard=None): ...

env **没有中央列表** —— 把包放到 ``robots/`` 下就行。这也是新增机器人
时用的机制 (见 :doc:`add_robot`)。

Planner 接口
------------

每个 planner 实现同一个很小的接口 (见 ``rpent.planner.base``):

- 接收已经渲染好的 ``system_prompt`` 和 ``user_message`` 字符串
  (CLI 在调用 ``solve`` 前从 env 的 prompt bundle 渲染得到)。
- 接收一个 ``toolkit`` (通过 ``get_tools_spec()`` 暴露 tool schema，
  通过 ``execute_tool(name, input_dict)`` 分发调用)。
- 驱动 tool-calling 循环。
- 把每个 tool 返回值以多模态上下文喂回。
- 遇到 ``finish`` 或触达上限时终止。

抽象就这些。三个内置 planner 只在 *如何满足契约* 上不同 —— 用户视角
见 :doc:`../usage/configure_planner`，源码见
``rpent/planner/api_loop.py`` / ``claude_code.py`` / ``codex.py``。

Toolkit 接口
------------

一个 toolkit (``rpent.tools.toolkit.Toolkit``) 持有：

- 一个 **primitive driver** —— 一个普通 Python 对象，持有 env
  client、VLA client 和任何 per-run 状态。LLM 能调的每个工具对应
  它的一个方法。
- 一组 **tool schema** (Anthropic 形状：``name``、``description``、
  ``input_schema``)，通过 ``self.add_tool(name, spec, handler)``
  注册。
- 每步的 **状态 dump** —— 每个 primitive tool 跑完后重新渲染世界，
  这样下一次 ``view_driver_state`` 看到的就是动作后的状态。

基类还处理 video 录制 (``episode.mp4``) 与 dashboard 事件流。
新增 env 的 ``toolkit.py`` 继承此基类并注册该 env 暴露的工具。

传输层
------

内置支持两种编码，通过 server 端 ``--transport {http,socket}``
(默认 ``http``) 选择，client 端由 ``--env-endpoint`` /
``--vla-endpoint`` 里的 protocol 前缀对应：

- **HTTP** (``rpent.utils.http_rpc``) —— JSON body 走
  ``POST /call``，方便做标准负载均衡，也方便跨语言 client。
  Numpy 数组在 wire 上带标签 ``{"__ndarray__": <base64>, "dtype": ..., "shape": [...]}``。
- **Pickle-framed socket RPC** (``rpent.utils.socket_rpc``) ——
  适合历史堆叠的嵌套 numpy dict 和宽泛、形状多变的载荷 (JSON 重编码
  在这种情况下太浪费)。

Server 端继承 :class:`rpent.utils.rpc.RpcFacade` 并实现
``_dispatch(method, args, kwargs)`` 即可；base 负责 shutdown、healthz、
transport 绑定、感知父进程死亡、以及干净收尾。新增一个传输只需要实现
两个方法的 ``RpcClient`` 接口 (``call(method, args, kwargs, timeout_s)``);
toolkit 和 planner 不用动。

Dashboard (可选)
----------------

``rpent/dashboard/`` 是一个 FastAPI app 加一份静态前端。
开了 ``--dashboard`` 时，``rpent/cli/main.py`` 会把它绑在
``--dashboard-host:--dashboard-port`` 上 (默认 localhost，随机端口)，
先起 launcher 页面选配置，然后 stream:

- Agent 的 reasoning token (SSE)。
- 实时相机 / Pi0.5 叠加帧。
- 动作时间线。
- 结束时的剪辑回放。

.. note::

   Dashboard 是纯观察性的，永远不影响循环，所以 dashboard 内部出错也
   不会拖垮一次 run。

下一步
------

- 新增机器人? —— :doc:`add_robot`。
- 新增 VLA / action primitive? —— :doc:`add_primitive`。
- 想了解 memory 的设计与接入点? —— :doc:`memory`。
- 需要完整的扩展 checklist? —— :doc:`add_robot`。
