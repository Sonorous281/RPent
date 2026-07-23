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
让智能体通过与物理世界的递归交互持续进化。它不预设某一个基础模型，而是把
感知、推理、记忆、执行、自我进化这些异构能力融合到一个统一的物理智能体中。
凭借持续的交互、反思与适应，智能体能够习得新能力，超越其初始设计。

RPent 建立在三条核心设计原则之上。

**服务化。** 各项能力都以可复用的服务形式部署，比如 VLA 策略服务、环境
服务、记忆语料库，每一个都能独立扩缩容、重启或替换。

**标准化。** 所有服务通过统一的接口连接：一套 ``Planner`` 协议、一套通用的
``Toolkit`` API、一套共享的 RPC 底座。异构的能力模块因此无需胶水代码即可组合。

**可组合。** 服务可以灵活拼装成各式各样的物理智能体。换一个 planner、混用来自
不同 VLA 的动作 primitive、接入新仿真器、加上记忆，都不必改动其余部分。

这三条原则让 RPent 跳出传统机器人控制框架的范畴，成为一套面向物理世界的
智能体基础设施：智能不只是被部署，还在被持续地构建、扩展和进化。

下面几节讲清楚这套架构是怎么落地的：三个进程各自负责什么、彼此如何通信，
以及代码在 ``rpent/`` 和 ``robots/`` 下如何组织。

关键特性
--------

以下是这套架构围绕的几点框架级承诺，后面各节会展开每一项的实现。

**LLM 全程在环。** LLM 不做微调，它完全靠调用工具（``pi0_pick``、
``move_to``、``rotate_wrist``、``back_project``、``finish`` 等）来驱动机器人。
每个工具的返回都作为多模态上下文（文本加渲染图）喂回去，让模型基于它实际
看到的画面来推理。

**三进程架构。** 系统由三个独立进程组成：Agent 进程运行 LLM planner 和
toolkit，不依赖 GPU；``env_server`` 负责仿真与 EGL 渲染；``vla_server`` 持有
GPU 上的策略权重。三者用轻量 RPC 串在一起，任何一个重量级进程都可以单独重启、
迁到另一张 GPU，或指向远程主机。

**可插拔的推理大脑。** 决策大脑用一个 ``--planner`` 参数就能替换，不必改动
工具或 prompt，内置三种可选：

- ``api``：基于 `pydantic-ai <https://ai.pydantic.dev/>`_ 的工具调用循环，
  与具体 provider 无关（兼容 Anthropic、OpenAI 及 OpenAI 兼容接口），
  自带 prompt 缓存和历史图片剪枝。
- ``claude_code``：`Claude Agent SDK
  <https://docs.claude.com/en/api/agent-sdk/overview>`_，把 toolkit 暴露成一个
  进程内的 MCP server。
- ``codex``：OpenAI Codex SDK，通过一个 HTTP MCP server 桥接到 toolkit。

**一份契约，多种环境。** env 与 vla 的进程拆分是一份通用契约。LIBERO（Pi0.5
走 HTTP）是已发布的参考实现，RoboCasa（RLDX-1 走 socket RPC）仍在开发中。
接入新环境只需实现同一套接口，随环境观测形状变化的只有底层的传输编码。

**实时 dashboard。** 加上 ``--dashboard`` 会启动一个本地 FastAPI 监控页，实时
展示智能体的推理过程、相机与 Pi0 视图、动作时间线和回放片段，界面支持中英双语
（``--dashboard-language {en, zh-cn}``）。

**加一个环境只需把包放进硬盘。** 没有中央注册表要改，详见 :doc:`add_robot`。

运行时循环
----------

一次运行就是一段 LLM 全程在环的循环：

1. LLM 分析任务，调用一个工具，比如 ``pi0_pick``。
2. 工具背后的 primitive driver 向 ``vla_server`` 请求一段动作 chunk（``predict``）。
3. ``env_server`` 执行这段 chunk（LIBERO 用 ``chunk_step``，RoboCasa 逐步 ``step``）。
4. 环境渲染出新的观测和相机画面。
5. 结果组装成文本加图像的内容块，喂回给 LLM 进入下一轮。

当 LLM 调用 ``finish``（结果为 ``success``、``failure`` 或 ``stuck``），或者
触达 ``--max-turns`` / ``--max-episode-steps`` 上限时，循环结束。

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
再交给推理循环。它先拉起 ``env_server`` 和 ``vla_server`` 两个子进程，等它们
就绪后，为所选环境构造 toolkit，并按参数选定的 planner 构造决策大脑，最后跑起
工具调用循环，把结果落盘。日常会用到的命令行参数在 :doc:`../quickstart` 里介绍。

Runner 有意保持轻薄：跟环境有关的东西都在 ``robots/<env>/`` 下，跟大脑有关的
都在 ``rpent/planner/`` 下。

Env 侧的注册表
--------------

``rpent/envs/base.py`` 维护一个以环境名为键的惰性注册表。传入 ``--env myenv``
时，它会按需 import ``robots.myenv`` 包，并调用包里暴露的两个工厂：

.. code-block:: python

   # robots/myenv/__init__.py
   def get_env_spec() -> EnvSpec: ...
   def get_toolkit(*, primitives_kwargs, video_path=None, dashboard=None): ...

环境没有中央列表，把包放进 ``robots/`` 下就够了。新增机器人用的也是这个机制
（见 :doc:`add_robot`）。

Planner 接口
------------

每个 planner 都实现同一个很小的接口（见 ``rpent.planner.base``）：

- 接收已经渲染好的 ``system_prompt`` 和 ``user_message`` 字符串。CLI 会在调用
  ``solve`` 之前，从环境的 prompt bundle 渲染出这两段文本。
- 接收一个 toolkit：工具的 schema 由 ``get_tools_spec()`` 暴露，调用则通过
  ``execute_tool(name, input_dict)`` 分发。
- 驱动工具调用循环。
- 把每个工具的返回值作为多模态上下文喂回去。
- 遇到 ``finish`` 或触达上限时终止。

抽象就这么多。三个内置 planner 的区别只在于各自怎么满足这份契约。用户视角的
介绍见 :doc:`../usage/configure_planner`，源码则在 ``rpent/planner/`` 下对应的
三个文件里。

Toolkit 接口
------------

一个 toolkit（``rpent.tools.toolkit.Toolkit``）持有三样东西：

- 一个 primitive driver。它是普通的 Python 对象，握着 env 客户端、VLA 客户端和
  本次运行的各种状态。LLM 能调的每个工具，都对应它上面的一个方法。
- 一组工具 schema，采用 Anthropic 的形状（``name``、``description``、
  ``input_schema``），通过 ``self.add_tool(name, spec, handler)`` 注册。
- 每一步的状态 dump。每个 primitive 工具跑完后都会重新渲染世界，这样下一次
  ``view_driver_state`` 看到的就是动作之后的状态。

基类还负责录制视频（``episode.mp4``）和 dashboard 的事件流。新增环境的
``toolkit.py`` 继承这个基类，注册该环境要暴露的工具即可。

传输层
------

内置支持两种传输编码，在服务端用 ``--transport {http,socket}`` 选择（默认
``http``），客户端则由 ``--env-endpoint`` / ``--vla-endpoint`` 里的协议前缀对应。

- **HTTP** 编码在 ``rpent.utils.http_rpc`` 中实现：JSON 请求体走 ``POST /call``，
  便于套用标准的负载均衡，也便于跨语言客户端接入。Numpy 数组在传输时会带上标签
  ``{"__ndarray__": <base64>, "dtype": ..., "shape": [...]}``。
- **Socket RPC** 编码在 ``rpent.utils.socket_rpc`` 中实现，采用 pickle 分帧：
  适合历史堆叠的嵌套 numpy dict，以及那些又宽又不规则、用 JSON 重编码太浪费的载荷。

服务端只要继承 :class:`rpent.utils.rpc.RpcFacade` 并实现
``_dispatch(method, args, kwargs)``；关闭、健康检查、传输绑定、感知父进程退出、
干净收尾这些都由基类兜底。想新增一种传输，只需实现 ``RpcClient`` 的两个方法，
toolkit 和 planner 都不用动。

Dashboard（可选）
-----------------

``rpent/dashboard/`` 是一个 FastAPI 应用加一份静态前端。开了 ``--dashboard``
时，``rpent/cli/main.py`` 会把它绑定到指定的主机和端口（默认本地、随机端口），
先弹出一个选配置的启动页，随后开始推送：

- 智能体的推理 token（走 SSE）。
- 实时的相机与 Pi0.5 叠加画面。
- 动作时间线。
- 结束时的回放片段。

.. note::

   Dashboard 是纯观察性的，永远不影响循环，所以即便它内部出错，也不会拖垮
   一次运行。

下一步
------

- 想新增机器人？见 :doc:`add_robot`。
- 想新增一个 VLA 或动作 primitive？见 :doc:`add_primitive`。
- 想了解记忆的设计和接入点？见 :doc:`memory`。
- 想要完整的扩展清单？见 :doc:`add_robot`。
