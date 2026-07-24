RoboCasa
========

.. note::

   RoboCasa 支持**尚在开发中**，暂不可用。下文描述的是*计划中*的集成方案，
   仓库里目前还没有 ``robots/robocasa/`` 包。当前状态请见功能矩阵。

`RoboCasa <https://robocasa.ai>`_ 是厨房尺度、长时序的操作环境。在 RPent 里，
它将由 **RLDX-1** 这个 VLA 策略驱动，走 pickle 分帧的 socket RPC 而非 LIBERO
所用的 HTTP。这是因为 RLDX 的观测是历史堆叠的嵌套 numpy dict，socket 能天然
承载，换成 HTTP 反倒要额外设计传输格式。

可用任务家族
------------

RoboCasa 覆盖了标准的厨房基准任务：

- ``PickPlace*``：把物体从起始位置搬到目标位置，比如从操作台到橱柜、从水槽到操作台。
- ``Open*`` 和 ``Close*``：开合橱柜门、抽屉和家电。
- ``TurnOn*`` 和 ``TurnOff*``：操作灶台旋钮、微波炉按钮、水壶开关这类开关。

具体有哪些任务取决于 RoboCasa 的版本，完整列表以
`RoboCasa <https://robocasa.ai>`_ 上游为准。

Toolkit 与 LIBERO 的差异
------------------------

RoboCasa toolkit 的工具 *构成* 和 LIBERO 一样，都是一次 primitive 调用、
一次状态查看、一次 ``finish``，但有两点是它特有的。

- **Env 侧的辅助方法。** 抓取检测和动作组装都需要一个在运行的仿真环境，所以
  它们做成了 env_server 的 RPC。这样一来，agent 侧的 skill 会同时握着两个
  客户端：env 客户端负责渲染和步进，模型客户端负责 RLDX-1 推理。理由参见
  :doc:`../development/add_robot`。
- **观测形状。** RLDX-1 看到的是 3 路相机的视频张量 ``(1, T, H, W, 3)``，
  按历史长度 ``T`` 堆叠，再加上 ``state.*``、标注，以及 session 和
  reset_memory 字段。
