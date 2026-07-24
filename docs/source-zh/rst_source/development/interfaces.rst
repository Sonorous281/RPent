核心接口
========

扩展 RPent 时，你要对接的就是下面这几个小而稳定的接口：加一个新环境、
写一个新 planner，或接入一种新传输，各自实现对应的接口即可。本页把它们
集中在一处。手把手的操作步骤见 :doc:`add_robot` 和 :doc:`add_primitive`,
整体设计见 :doc:`architecture`。

环境入口
--------

传入 ``--env myenv`` 时，``rpent/envs/base.py`` 会按需 import
``robots.myenv`` 包，并调用包里暴露的两个工厂：

.. code-block:: python

   # robots/myenv/__init__.py
   def get_env_spec() -> EnvSpec: ...
   def get_toolkit(*, primitives_kwargs, video_path=None, dashboard=None): ...

新增环境不用在别处登记，把包放进 ``robots/`` 下就会被发现。``get_env_spec`` 返回环境的
身份和 prompt bundle，``get_toolkit`` 构造 planner 驱动的 toolkit。

Planner 接口
------------

每个 planner 都实现同一个很小的接口(见 ``rpent.planner.base``):

- 接收已经渲染好的 ``system_prompt`` 和 ``user_message`` 字符串。CLI 会在调用
  ``solve`` 之前，从环境的 prompt bundle 渲染出这两段文本。
- 接收一个 toolkit:工具的 schema 由 ``get_tools_spec()`` 暴露，调用则通过
  ``execute_tool(name, input_dict)`` 分发。
- 驱动工具调用循环。
- 把每个工具的返回值作为多模态上下文喂回去。
- 遇到 ``finish`` 或触达上限时终止。

抽象就这么多。三个内置 planner 的区别只在于各自怎么满足这份契约。用户视角的
介绍见 :doc:`../usage/configure_planner`，源码则在 ``rpent/planner/`` 下对应的
三个文件里。

Toolkit 接口
------------

一个 toolkit(``rpent.tools.toolkit.Toolkit``)持有三样东西：

- 一个 primitive driver。它是普通的 Python 对象，握着 env 客户端、VLA 客户端和
  本次运行的各种状态。LLM 能调的每个工具，都对应它上面的一个方法。
- 一组工具 schema，采用 Anthropic 的形状(``name``、``description``、
  ``input_schema``)，通过 ``self.add_tool(name, spec, handler)`` 注册。
- 每一步的状态 dump。每个 primitive 工具跑完后都会重新渲染世界，这样下一次
  ``view_driver_state`` 看到的就是动作之后的状态。

基类还负责录制视频(``episode.mp4``)和 dashboard 的事件流。新增环境的
``toolkit.py`` 继承这个基类，注册该环境要暴露的工具即可。

传输层
------

内置支持两种传输编码，在服务端用 ``--transport {http,socket}`` 选择(默认
``http``)，客户端则由 ``--env-endpoint`` / ``--vla-endpoint`` 里的协议前缀对应。

- **HTTP** 编码在 ``rpent.utils.http_rpc`` 中实现：JSON 请求体走 ``POST /call``,
  便于套用标准的负载均衡，也便于跨语言客户端接入。Numpy 数组在传输时会带上标签
  ``{"__ndarray__": <base64>, "dtype": ..., "shape": [...]}``。
- **Socket RPC** 编码在 ``rpent.utils.socket_rpc`` 中实现，采用 pickle 分帧：
  适合历史堆叠的嵌套 numpy dict，以及那些又宽又不规则、用 JSON 重编码太浪费的载荷。

服务端只要继承 :class:`rpent.utils.rpc.RpcFacade` 并实现
``_dispatch(method, args, kwargs)``；关闭、健康检查、传输绑定、感知父进程退出、
干净收尾这些都由基类兜底。想新增一种传输，只需实现 ``RpcClient`` 的两个方法，
toolkit 和 planner 都不用动。
