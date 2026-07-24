Action Primitives
=================

planner 决定 *做什么*，而 **action primitive** 决定 *怎么做*。所谓 primitive，
就是把一次工具调用（比如 ``pi0_pick``、``move_to``、``set_gripper``）落地成
一段能直接交给环境执行的动作。

RPent 内置支持两大类 primitive。

- **VLA 策略**，即视觉-语言-动作模型。它跑在专门的 vla_server 进程里，
  把 GPU 权重和物理引擎隔开，toolkit 通过每个环境各自的模型客户端来调用它。
  Pi0.5（用于 LIBERO）和 RLDX-1（用于 RoboCasa）都属于这一类。
- **脚本化 primitive**，即确定性的运动，比如 ``move_to``、``rotate_wrist``、
  ``release``、``back_project``。它们不需要 VLA 权重，跑在 agent 侧，通过
  env_server 的 RPC 调用。

每种机器人具体怎么配置，包括用哪个 VLA、checkpoint 放在哪、暴露哪些工具，都参见对应的
环境页面：:doc:`libero`、:doc:`robocasa`、:doc:`franka`、:doc:`so101`。

不同环境用哪个 VLA
------------------

.. list-table::
   :header-rows: 1
   :widths: 25 25 25 25

   * - Environment / 机器人
     - 默认 VLA
     - 传输协议
     - Server
   * - LIBERO (仿真)
     - Pi0.5
     - HTTP 或 socket RPC (``--transport``)
     - ``robots/libero/vla_server.py``
   * - RoboCasa (仿真)
     - RLDX-1
     - pickle-framed socket RPC
     - ``robots/robocasa/vla_server.py`` *(规划中)*
   * - Franka (真机)
     - Pi0.5 或 RLDX-1 (依任务而定)
     - HTTP 或 socket
     - ``robots/franka/vla_server.py`` *(规划中)*
   * - SO-101 (真机)
     - RLDX-1 (依任务而定)
     - socket RPC
     - ``robots/so101/vla_server.py`` *(规划中)*

所有 VLA server 都用同一套 ``predict`` 和 ``healthz`` 方法，同时支持 HTTP（走
JSON）和 socket（pickle 分帧）两种传输，用 ``--transport {http,socket}`` 选择，
默认是 ``http``。这么设计的理由参见 :doc:`../development/add_robot`。

复用一个已在运行的 VLA server
-----------------------------

每一个 VLA server 都设计成 **可跨 run 复用**。用 ``--vla-endpoint``
指向已在跑的实例，而不是每次都启动新实例：

.. code-block:: bash

   rpent --env libero --vla-endpoint http://localhost:8000 \
     --suite libero_object_swap --task 2 --seed 0 --planner api \
     --model anthropic:claude-opus-4-8

``--vla-endpoint`` 接受 ``[protocol://]host:port`` 格式，其中 protocol 可以是
``http`` 或 ``socket``，默认 ``http``。复用已有 env_server 的 ``--env-endpoint``
也遵循同样的规则。

新增全新的 primitive 家族
-------------------------

如果要接入的东西既不是 VLA、也不是脚本化运动，比如一个世界动作模型
（WAM）、扩散规划器或 MPC primitive，参见 :doc:`../development/add_primitive`。
