添加 Action Primitive
=====================

RPent 里的 *action primitive*，就是把一次工具调用落地成环境可执行动作的那个
东西。它可以是一个学出来的策略，比如 VLA、WAM、扩散规划器，也可以是一段脚本化
的例程，比如 ``move_to``、``open_gripper``。本页说明这两类各自怎么加。

两种 primitive 形态
-------------------

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - 类别
     - 跑在哪里
     - 例子
   * - **基于模型的**
       (VLA、WAM、扩散等)
     - 自己的进程 vla_server，通过 toolkit 持有的模型客户端调用。
     - Pi0.5 (LIBERO)、RLDX-1 (RoboCasa)
   * - **脚本化的**
       (运动学或启发式)
     - agent 进程内，需要运动学时可能走一次 driver 侧 RPC，没有模型权重。
     - ``move_to``、``rotate_wrist``、``release``、
       ``back_project``

两种形态呈现给 LLM 的方式完全一样：一份 tool schema、一个 primitive-driver
方法、调用之后一次状态 dump。区别只在于 *方法内部做什么*。

添加一个脚本化 primitive
------------------------

脚本化 primitive 最容易加。步骤如下。

1. **在 primitive driver 上加一个方法。** 在你环境的 primitive driver 类
   (比如 ``LiberoPrimitives``、``MyRobotPrimitives``) 上加一个方法。它接收工具的
   kwargs，做实际的事情，通常是一次或多次 ``self._env.step(...)``，最后返回一个
   小小的 ``dict`` 日志。

   .. code-block:: python

      def open_drawer(self, dx: float = 0.15) -> dict:
          # 保持夹爪闭合, 沿 -x 方向后拉 dx 米。
          for _ in range(N):
              self._env.step(build_open_drawer_chunk(dx))
          return {"ok": True, "dx": dx}

2. **写 tool schema。** 在 ``toolkit.py`` 的 ``TOOLS_SPEC`` 里加一条：

   .. code-block:: python

      {
          "name": "open_drawer",
          "description": "Pull the currently-grasped drawer handle "
                         "backwards by ``dx`` meters.",
          "input_schema": {
              "type": "object",
              "properties": {"dx": {"type": "number"}},
              "required": [],
          },
      }

3. **在 toolkit 中注册。** 让 tool 走 toolkit 的 ``_step`` 辅助函数，
   这样跑完后会自动重新渲染状态：

   .. code-block:: python

      self.add_tool("open_drawer", OPEN_DRAWER_SPEC,
                    lambda **kw: self._step("open_drawer", **kw))

到这一步，``api``、``claude_code``、``codex`` 三种 planner 就都能调用它了，
其余什么都不用改。

添加一个 VLA (或其他基于模型的 primitive)
-----------------------------------------

基于模型的 primitive 要多搭一点脚手架，因为模型跑在自己的进程里。步骤如下。

1. **写一个 ``vla_server.py``。** 它只持有模型权重和 CUDA 上下文，继承
   :class:`rpent.utils.rpc.RpcFacade`，通过 ``_dispatch`` 暴露模型方法，比如
   ``predict``。

   - 默认走 **HTTP**，也就是 JSON over ``POST /call``，适合扁平的
     ``image + state`` 数据，这是 LIBERO 和 Pi0.5 的模式。
   - 当观测是带历史堆叠的嵌套 numpy dict 时，切到 **socket RPC**
     (用 ``--transport socket``)，省掉 JSON 重编码的开销。

   ``RpcFacade.serve`` 会负责传输绑定、``healthz``、``shutdown``、感知父进程
   退出这些杂事，你只需要写模型相关的方法。

2. **写一个模型客户端。** 一个小类，包住一个 :class:`rpent.utils.rpc.RpcClient`
   (``HttpRpcClient`` 或 ``SocketRpcClient``)，对外暴露模型的业务 API。可以参考
   LIBERO 用的 ``rpent.utils.vla_client.VLAClient``。

3. **在 primitive driver 上加一个方法。** 在环境的 primitive driver 类里调用
   模型客户端，把返回的 chunk 交给 env，再返回一个日志 dict:

   .. code-block:: python

      def mymodel_pick(self, target: str) -> dict:
          env_obs = self._env.raw_obs()
          env_obs["task_descriptions"] = f"pick {target}"
          chunk, _ = self._model.predict_action_batch(env_obs, mode="eval")
          self._env.chunk_step(chunk)
          return {"model": "mymodel", "target": target}

4. **加上 tool schema** 并在 toolkit 里注册，做法和脚本化那一节完全一样。

5. **在 ``__init__.py`` 里串起来。** 环境的 ``get_toolkit`` 用正确的
   ``primitives_kwargs`` 构造 toolkit:

   .. code-block:: python

      def get_toolkit(*, primitives_kwargs, video_path=None, dashboard=None):
          from robots.myrobot.toolkit import MyRobotToolkit
          return MyRobotToolkit(
              primitives_kwargs=primitives_kwargs,
              video_path=video_path,
              dashboard=dashboard,
          )

   ``rpent/cli/main.py`` 会传入 ``{"env": MyRobotEnvClient(...),
   "model": MyModelClient(...)}``。

跨 run 复用同一个 vla_server
----------------------------

模型 server 启动很贵，加载权重是大头。Runner 支持指向一个已经在跑的实例：

.. code-block:: bash

   rpent --env libero --vla-endpoint http://vla-host:8000 ...

把你的 vla_server 设计成与任务无关：LIBERO 参考实现在启动时只加载一次模型，
之后每次 ``predict`` 调用都彼此独立，因此不需要 reset RPC，一个进程就能安全地
连续服务很多次运行。

新 primitive 的设计原则
-----------------------

- **工具描述意图，而不是动作。** 好的工具名叫 ``pi0_pick``，而不是
  ``execute_action_chunk_of_length_20``。LLM 是靠名字来挑工具的，名字得能自解释。
- **每个工具结束时都要 dump 状态。** 下一轮依赖这份 dump 来反映动作之后的世界，
  所以别在渲染完成之前就让 primitive 提前返回。
- **返回一个小 dict。** 工具的返回值会以文本形式喂回 LLM，控制在几百字节以内。
  图像、深度、``states.json`` 这类大块数据走 state dump，以图像内容块的形式回传。
- **护栏属于 env_server，不属于 toolkit。** LLM 会用任意参数调任意工具，所以
  工作空间边界和安全钳位必须在 driver 侧强制执行。

超越 VLA
--------

同样的模式可以扩展到非 VLA 的模型 primitive。

- **世界动作模型 (WAM)** 基于想象做 rollout，产出一个计划交给 env 执行。接法和
  VLA 一模一样：自己的进程、自己的客户端。
- **扩散规划器或 MPC** 形态也一样，只是工具返回的"动作"可能是一整段轨迹而非
  单个 chunk，由 env_server 一步步走完。
- **多个 primitive 共享一个 server**：一个 vla_server 可以承载多个模型，
  由工具通过 ``predict`` 的 ``model`` 参数选调哪个 head。

无论形态如何，框架的契约始终不变：模型进程交给模型客户端，客户端交给
primitive driver 方法，再配上 tool schema，最后用 ``Toolkit.add_tool`` 注册。
