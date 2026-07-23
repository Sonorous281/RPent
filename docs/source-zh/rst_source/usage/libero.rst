LIBERO
======

`LIBERO <https://libero-project.github.io/>`_ 是 RPent 默认的 environment:
一个基于 MuJoCo/robosuite 的桌面操作基准，包含四个套件
(``libero_object``、``libero_goal``、``libero_spatial``、``libero_10``)
和三个变体 (``standard``、``pro``、``plus``)。默认 VLA 是 **Pi0.5**，
由 ``robots/libero/vla_server.py`` 通过 HTTP 提供服务。

VLA 配置
--------

Pi0.5 只需要一件事：磁盘上的 checkpoint。通过 ``PI05_CHECKPOINT_PATH``
指向它：

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

推荐的 SFT checkpoint 可以从 HuggingFace 下载：
`RLinf-Pi05-LIBERO-130-fullshot-SFT
<https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_。

任务选择
--------

每一次 LIBERO 运行都要指定：

- ``--suite`` —— 四个套件之一，可选地带上变体后缀 (见下)。例：
  ``libero_object_task``、``libero_object_swap``、
  ``libero_goal_lan``、``libero_spatial_task``、
  ``libero_10_swap``。
- ``--task`` —— 套件内的任务索引。
- ``--seed`` —— environment 种子。
- ``--libero-type`` —— LIBERO 变体：``standard`` | ``pro`` |
  ``plus``。不填时 RPent 会读环境变量 ``LIBERO_TYPE`` (默认 ``pro``)。

套件 × 变体一览
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - 套件
     - 变体
     - 用途
   * - ``libero_object``
     - ``_task`` / ``_swap`` / ``_lan``
     - 面向物体的任务，支持目标 swap 或语言扰动。
   * - ``libero_goal``
     - ``_task`` / ``_swap`` / ``_lan``
     - 目标条件任务，支持 swap / 语言扰动。
   * - ``libero_spatial``
     - ``_task`` / ``_lan``
     - 空间关系任务。
   * - ``libero_10``
     - ``_task`` / ``_swap`` / ``_lan``
     - 长时序的 LIBERO-10 套件。

最小命令
--------

.. code-block:: bash

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0

   rpent --env libero \
     --suite libero_object_swap --task 2 --seed 0 \
     --planner api --model anthropic:claude-opus-4-8 \
     --max-tokens 8192

进程分工
--------

- **env_server** (``robots/libero/env_server.py``) —— 持有 LIBERO
  MuJoCo env 与 EGL 渲染。通过 RPC 传输 (默认 HTTP；加
  ``--transport socket`` 走 pickle-framed socket) 对外暴露
  ``reset``、``step``、``chunk_step``、``render_camera(camera_name="agentview")``、
  ``get_camera_meta``、``cached_image``…
- **vla_server** (``robots/libero/vla_server.py``) —— 持有 Pi0.5
  权重，通过同一套 RPC 传输 (HTTP 或 socket) 暴露 ``predict``。
- **Toolkit** (``robots/libero/toolkit.py``) —— 定义 LLM 能调的工具：
  ``pi0_pick`` (交给 Pi0.5)、``move_to``、``rotate_wrist``、
  ``back_project``、``view_driver_state``、``finish``…

Planner 能看到的工具
--------------------

LIBERO toolkit 默认暴露：

- ``pi0_pick(prompt)`` —— 调用 Pi0.5 生成一次由 ``prompt`` (自然语言
  抓取指令) 驱动的抓取动作块。
- ``pi0_doubled(prompt)`` —— 调用 Pi0.5 生成一次由 ``prompt`` 驱动的
  非抓取接触动作块 (如拧旋钮、开关炉灶、短推)。
- ``move_to(xyz)`` —— 脚本化 Cartesian 运动，移动到绝对世界坐标系目标
  ``[x, y, z]`` (单位：米) (确定性，不走 VLA)。
- ``move_pose(xyz)`` —— 脚本化 Cartesian 运动，同时协同调整位置与腕部
  姿态 (pitch + yaw)；用于穿入柜门前 / 低矮层架等姿态，避免解耦伺服卡死。
- ``rotate_wrist(target_yaw / delta_yaw)`` —— 脚本化的腕关节旋转
  (绕世界 Z 轴)；给绝对 ``target_yaw`` 或相对 ``delta_yaw`` (弧度)。
- ``rotate_pitch(target_pitch / delta_pitch)`` —— 脚本化的夹爪俯仰
  (绕世界 X 轴)；给绝对 ``target_pitch`` 或相对 ``delta_pitch`` (弧度)。
- ``release()`` —— 打开夹爪。
- ``set_gripper(gripper, steps)`` —— 保持当前姿态，驱动夹爪指令
  ``steps`` 个 env step (如搬运途中收紧抓握)。
- ``back_project(row, col)`` —— 把图像像素 (``row`` 0 = 顶部，
  ``col`` 0 = 左侧) 反投影到世界坐标 3D 点。
- ``segment(prompt)`` —— 可选的分割辅助工具，在已有图像 artifact 上
  定位物体 (未配置服务时回退到人工定位)。
- ``view_driver_state()`` —— 强制刷新一次状态 dump (图像、深度、
  camera meta、``states.json``)。
- ``view_camera_meta(camera)`` —— 读取相机标定元数据 (``agentview``
  或 ``wrist``) 用于定位。
- ``finish(status, summary)`` —— 结束当前 episode; ``status`` 为
  ``success`` / ``failure`` / ``stuck``, ``summary`` 为简短的自然语言
  总结 (两者都必填)。

每个工具跑完后都会重新渲染世界，所以下一轮 agent 上下文反映的是
动作后的状态。

Dashboard
---------

给 LIBERO 运行加上 ``--dashboard`` 打开本地监控页：

.. code-block:: bash

   rpent --env libero --dashboard \
     --suite libero_goal_task --task 1 --seed 0 --planner claude_code

Dashboard streams reasoning、agentview + 腕部相机 + Pi0.5 叠加视图，
以及动作时间线。用 ``--dashboard-language zh-cn`` 切换中文 UI。

自带 VLA
--------

如果你有一个与 LIBERO 兼容、但不是 Pi0.5 的 VLA，可以在不动 env 的
情况下把 model client 换掉：

1. 写一个新的 ``vla_server.py``，暴露相同的 ``predict`` RPC 契约
   (http 或 socket 皆可)。
2. 用 ``--vla-endpoint [protocol://]host:port`` 指向它。
3. 如果 tool surface 要变 (比如 ``pi0_pick`` 改成 ``mymodel_pick``)，
   相应更新 ``robots/libero/toolkit.py``。

完整流程见 :doc:`../development/add_primitive`。
