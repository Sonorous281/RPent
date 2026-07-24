添加新机器人
============

本指南说明，把一个新的物理或仿真机器人接入 RPent 的 LLM 在环 runner 时，需要
写哪些东西。请把 ``robots/libero/`` 当作完整的参考实例。各模块要实现的接口契约，
可先看 :doc:`interfaces` 总览。

RPent 把一个 env 拆成两个进程。

- **Agent 侧**，即 ``robots/<env>/``，跑在 agent 进程内，提供工具 schema、
  primitive driver 逻辑和 prompt。
- **Driver 侧**，即 ``robots/<env>/env_server.py``，持有重量级的仿真器或机器人，
  通过 :class:`rpent.utils.rpc.RpcFacade` 对外暴露 env。它默认走 HTTP，加
  ``--transport socket`` 可以切到 pickle 分帧的 TCP 传输，适合观测形态偏大的场景。

两侧通过一个 ``EnvClient`` 类相连，agent 侧的每次方法调用，对应一次发往 driver 的 RPC。

VLA 模型跑在独立的进程里
------------------------

当一个 env 用到 VLA 策略，也就是那种读相机观测、输出动作的学习模型时，这个模型
会跑在**第三个独立进程**里，绝不塞进 env_server。

- **VLA 侧**，即 ``robots/<env>/vla_server.py``，只持有 VLA 策略，也就是 GPU 上的
  模型。它通过自己的 RPC 或 HTTP 端点暴露一个 ``predict`` 推理 RPC，不 import
  任何仿真器。
- toolkit 除了 ``EnvClient`` 之外，还会接收一个模型客户端作为 ``model`` 参数。
  LIBERO 上的 Pi0.5 用 ``VLAClient``，RoboCasa 上的 RLDX-1 用 ``RLDXVLAClient``。
  两个客户端分别指向两个不同的 server 进程。

**为什么这个分离是强制的，而不是可选的。** 模型和仿真器在进程层面的需求是冲突
的：模型带着庞大的 GPU 权重、自己的 CUDA 上下文，以及 ``transformers``、``openpi``
这类重依赖；仿真器则是 MuJoCo、robosuite，还有绑在主线程上的 EGL 渲染。把两者塞进
同一个进程，会让它们的生命周期耦在一起，逼一个解释器同时满足两套依赖树，而且模型
一旦 OOM 就会连带拖垮仿真。拆开之后，任何一侧都能独立重启、扩容或指向远程主机，用
远程主机，用 ``--vla-endpoint host:port`` 就能复用一个已经在跑的模型 server。所以
每个 env 都必须守住这条线：env_server 持有仿真，vla_server 持有模型。

**传输协议可以随 env 变，但架构不能变。** LIBERO 默认让 env_server 和 vla_server
都走 HTTP。如果某个机器人的观测是历史堆叠的嵌套 numpy dict，可能更适合走 pickle
分帧的 socket（``--transport socket``），省掉 JSON 重编码的开销。两种传输通过
:class:`RpcFacade` 共用同一套 ``predict`` 和 ``env.*`` 方法接口。你按观测形态挑
编解码方式，但 env 与 vla 的进程分离始终保持一致。

**任何需要仿真 env 对象的逻辑，都留在 env_server 里。** 拿 RoboCasa 这样的 env
来说，来说，抓取检测、动作组装这些操作都需要一个活着的仿真 env，因此它们是 env_server
的 RPC，并不归 VLA server 管。于是 agent 侧的 skill 会同时握着两个客户端：env
客户端负责渲染和步进，模型客户端负责推理。

入口
----

新增名为 ``myenv`` 的 env 时，文件布局如下：

.. code-block:: text

   robots/myenv/
       __init__.py            # 入口: get_env_spec() 与 get_toolkit() 工厂
       env_client.py          # MyEnvClient, agent 侧 RPC 代理 (§1)
       prompt_bundle.py       # system() 与 user() prompt 工厂            (§2)
       toolkit.py             # MyEnvToolkit + primitives + tool schemas (§3)
       env_server.py          # driver 侧 facade + RPC server (§1)
       vla_server.py          # (可选) VLA 模型 server (§1)

``__init__.py`` 是这个包的入口。``rpent/envs/base.py`` 里的注册表会按需惰性 import
``robots.<name>``，再调用它的两个工厂函数：

.. code-block:: python

   # robots/myenv/__init__.py
   from rpent.envs.env_spec import EnvSpec
   from rpent.envs.prompt_bundle import PromptBundle
   from robots.myenv.prompt_bundle import system_prompt, user_prompt

   def get_env_spec() -> EnvSpec:
       return EnvSpec(name="myenv", prompts=PromptBundle(system=system_prompt, user=user_prompt))

   def get_toolkit(*, primitives_kwargs: dict[str, Any], video_path: str | None = None, dashboard: Any = None):
       from robots.myenv.toolkit import MyEnvToolkit
       return MyEnvToolkit(primitives_kwargs=primitives_kwargs, video_path=video_path, dashboard=dashboard)

整个注册流程就这么多。``_resolve_env(name)`` 通过
``importlib.import_module(f"robots.{name}")`` 动态加载，所以把包放到 ``robots/``
下就够了，没有中央列表需要维护。

下面三节分别说明上面引用的三个模块各自要写什么。

1. ``env_client.py`` + ``env_server.py``
-----------------------------------------

这两个文件构成 agent 和 driver 之间的桥梁：client 跑在 agent 进程内，把方法调用
转成 RPC；env_server 跑在 driver 进程内，应答这些调用。

1.1 Env client (agent 侧)
~~~~~~~~~~~~~~~~~~~~~~~~~

类里约定两个 gym 风格的方法（``reset`` 和 ``step``），其余按 env 需要增加。
LIBERO 就额外加了 ``chunk_step``、``render_camera``、``get_camera_meta``、
``cached_image`` 等。每个方法都通过
``RpcClient.call("<rpc-name>", args=..., kwargs=...)`` 转发，并各自设好 timeout。
方法名要保持稳定，因为 driver 侧的 dispatcher 是按名字匹配的。

.. code-block:: python

   class MyEnvClient:
       def __init__(self, client: RpcClient, *, return_all_frames: bool = False):
           self._client = client
           self.return_all_frames = return_all_frames

       def reset(self):
           return self._client.call("env.reset", timeout_s=120.0)

       def step(self, action):
           return self._client.call("env.step", args=(action,), timeout_s=60.0)
       # ... 根据 env 需要添加其他方法

1.2 Env server (driver 侧)
~~~~~~~~~~~~~~~~~~~~~~~~~~

在 driver 侧用一个 facade 类（比如 ``MyEnvFacade``）镜像 client 的 API。它继承
:class:`rpent.utils.rpc.RpcFacade`，实现 ``_dispatch(method, args, kwargs)`` 把
``env.*`` 路由到自己的方法，再用 ``self.serve(...)`` 起服务。方法接收的位置参数和
关键字参数要和 client 发来的一致，返回值必须可 pickle（用 numpy，不要 torch，
因为 agent 侧不 import torch）。

.. code-block:: python

   from rpent.utils.rpc import RpcFacade

   class MyEnvFacade(RpcFacade):
       def __init__(self, env, meta):
           super().__init__()
           self._env = env
           self._meta = meta

       def _dispatch(self, method, args, kwargs):
           if method.startswith("env."):
               return getattr(self, method[len("env."):])(*args, **kwargs)
           raise ValueError(f"unknown RPC method: {method!r}")

       def reset(self): ...
       def step(self, action): ...

   facade = MyEnvFacade(env, meta)
   facade.serve(transport="http", host=host, port=port)

``RpcFacade.serve`` 会负责传输绑定（http 或 socket）、``healthz`` 和 ``shutdown``
方法、感知父进程退出，以及干净收尾，你只需要写业务方法。

把新 env 接入 ``rpent/cli/main.py`` 目前需要三个具体步骤。第一，把 env 名加进
``--env`` 的 ``choices`` 列表。第二，仿照 ``_init_libero`` 写一个 ``_init_<env>(...)``
构建器，负责拉起 env 和 vla 守护进程并返回 ``primitives_kwargs``。第三，在
``_build_env_parser`` 里加上对应分支，因为它目前对任何非 ``libero`` 的 env 都会
``assert False``。

2. ``prompt_bundle.py``
-----------------------

定义两个 prompt 工厂 ``system_prompt()`` 和 ``user_prompt()``，并在 env 的
``__init__.py`` 里构造 ``PromptBundle(system=system_prompt, user=user_prompt)``，
见上面的入口一节。每个工厂返回一个有序的 ``dict[str, PromptNode]``，也就是带标题的
分节，由 ``PromptBundle.render`` 组装并填充。一份 prompt 服务所有 planner，包括
API loop、Claude Code 和 Codex：工具用裸名引用（比如 ``move_to``），只需说明一次
Claude Code 和 Codex SDK 会把它们命名空间化成 ``mcp__rpent__<name>``，不必再维护
CLI 和 API 两份拷贝。

.. code-block:: python

   # robots/myenv/prompt_bundle.py
   from rpent.context.prompt_utils import PromptNode
   from rpent.context.prompts import prompt as base_prompt
   from robots.myenv import prompts as myenv_prompt

   def system_prompt() -> dict[str, PromptNode]:
       return {
           "Intro": myenv_prompt.PREAMBLE,
           "Goal": myenv_prompt.GOAL,
           "Rules": myenv_prompt.RULES,
           "Workflow": myenv_prompt.WORKFLOW,
           "Environment": myenv_prompt.ENVIRONMENT,
           "Output": base_prompt.OUTPUT,
       }

   def user_prompt() -> dict[str, PromptNode]:
       return dict(base_prompt.USER)

你可以复用 ``rpent.context.prompts.prompt`` 里的共享分节（``OUTPUT``、``USER``），
也可以自己写。分节内容是普通字符串，或者 ``BulletList``、``Numbered``，其中的占位符
``{{suite}}``、``{{task}}``、``{{seed}}``、``{{output_dir}}``、``{{recipe_tag}}``
会在渲染时填充。

3. ``toolkit.py``
------------------

这个模块持有 LLM 能调用的一切：工具 schema、primitive driver、每步状态 dump，
以及 MCP allowlist。（LIBERO 因为历史原因把这些拆到了 ``tools.py`` 和 ``toolkit.py``
两个文件；新增 env 时全放在 ``toolkit.py`` 里也没问题。）

一个 toolkit 模块通常包含四部分。

**Primitive driver 类**, 比如 ``MyEnvPrimitives``, 是 toolkit 持有的 Python 对象。
它保存 ``EnvClient``、VLA 模型客户端和本次运行的各种状态，每个 primitive 工具
(``move_to``、``pi0_pick``、``release`` 等) 对应它上面的一个方法，返回一个 ``dict``
形式的日志。

**工具 schema 和 handler 辅助函数**，包括一个模块级的 ``TOOLS_SPEC`` 列表（采用
Anthropic 的形状，每条含 ``name``、``description``、``input_schema``），以及 toolkit
引用的 env 专属自由函数，比如 ``view_driver_state``、``back_project``。像 ``finish``
这样的通用工具定义在 ``rpent/tools/common.py`` 里，由基类 ``Toolkit`` 自动注册，
不必每个 env 重新定义一遍。

**每步状态 dump**，即 ``dump_state(driver, output_dir, step_idx, log)``，把 agent
之后会通过 ``view_*`` 工具读回的所有状态（图像、深度、JSON 状态、camera meta）
序列化到 ``output_dir``。

**Toolkit 类**，继承 ``rpent.tools.toolkit.Toolkit``。

- 在 ``__init__`` 里通过 ``init_primitives_clean`` 构建 primitive driver，它会清理
  过期的 ``images/`` 等，构造 primitives，并 dump 第 0 步。
- 用 ``self.add_tool(name, spec, handler)`` 注册每个工具。无状态的读取类工具
  (``view_driver_state``、``back_project`` 等) 直接绑定到模块级函数；primitive
  工具则走 ``_step(name, **kwargs)``，由它通过 ``getattr(self._primitives, name)``
  调用 driver 方法并重新渲染状态。
- override ``close()`` 来 flush agent 侧的产物，比如 LIBERO toolkit 就在这里保存
  agentview 的 MP4。

``primitives_kwargs`` 由 ``__init__.py`` 的 ``get_toolkit`` 转发进来，toolkit 把它
原样传给 primitive driver 的 ``__init__``，通常长这样:
``{"env": MyEnvClient(...), "model": VLAClient(...), ...}``。

值得遵循的约定
--------------

- ``output_dir`` 是本次运行的临时目录，由 runner 创建；所有产物（图像、深度、
  ``states.json``、transcript、``episode.mp4``）都写在里面。
- 工具 schema 采用 Anthropic 形状，含 ``name``、``description``、``input_schema``。
  每个用 ``self.add_tool(...)`` 注册的工具都会暴露给所有 planner。
- Driver 侧的返回值必须可 pickle，且不含 torch。
- 每个 primitive 工具执行后都要 dump 一份新的状态快照，这样下一次
  ``view_driver_state`` 看到的才是动作之后的世界。
- 把 ``dump_state`` 当作 agent 视角的"事实源"，任何新的模态（比如触觉、力）都从
  它这里走。

冒烟测试
--------

代码可以编译之后，最小的冒烟回路如下：

.. code-block:: bash

   PI05_CHECKPOINT_PATH=<path> ANTHROPIC_API_KEY=<key> \
     rpent --env myenv --suite <suite> --task <id> --seed 0 \
     --output-dir /tmp/myenv_smoke --planner api --model anthropic:claude-opus-4-8

期望的结果是：agent 完成 prompt 里的任务，并调用 ``finish``。查看
``<output_dir>/transcript_*.json`` 就能拿到运行结束时的总结。
