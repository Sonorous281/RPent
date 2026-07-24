LIBERO
======

`LIBERO <https://libero-project.github.io/>`_ 是 RPent 默认的环境，一个基于
MuJoCo 和 robosuite 的桌面操作基准。它包含四个套件（``libero_object``、
``libero_goal``、``libero_spatial``、``libero_10``）和三个变体（``standard``、
``pro``、``plus``）。默认的 VLA 是 **Pi0.5**，由 ``robots/libero/vla_server.py``
通过 HTTP 提供服务。

VLA 配置
--------

下载推荐的 SFT checkpoint
`RLinf-Pi05-LIBERO-130-fullshot-SFT
<https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_，
再通过 ``PI05_CHECKPOINT_PATH`` 指向它：

.. code-block:: bash

   hf download RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT \
     --local-dir /path/to/rlinf-pi05-libero-130-fullshot-sft

   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft

SAM3 配置
---------

每次 LIBERO 运行都默认启用 SAM 3.0 分割。可以用以下任一方式下载
``sam3.pt``，再通过 ``SAM3_CHECKPOINT_PATH`` 指定本地 checkpoint：

.. code-block:: bash

   # Hugging Face（需要先在模型页面申请访问权限）
   hf auth login
   hf download facebook/sam3 sam3.pt --local-dir /path/to/sam3

   # ModelScope（与上面的 Hugging Face 命令二选一）
   modelscope download --model facebook/sam3 sam3.pt --local_dir /path/to/sam3

   export SAM3_CHECKPOINT_PATH=/path/to/sam3/sam3.pt

SAM 3.0 checkpoint 可以从以下页面下载：
`Hugging Face: facebook/sam3 <https://huggingface.co/facebook/sam3>`_、
`ModelScope: facebook/sam3 <https://modelscope.cn/models/facebook/sam3>`_。

任务选择
--------

每一次 LIBERO 运行都要指定这几项：

- ``--suite`` 是四个套件之一，可以带上变体后缀（见下文），例如
  ``libero_object_task``、``libero_object_swap``、``libero_goal_lan``、
  ``libero_spatial_task``、``libero_10_swap``。
- ``--task`` 是套件内的任务索引。
- ``--seed`` 是环境种子。
- ``--libero-type`` 指定 LIBERO 变体，取 ``standard`` | ``pro`` | ``plus``。
  不填时 RPent 会读环境变量 ``LIBERO_TYPE``，默认 ``pro``。

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

- **env_server**，也就是 ``robots/libero/env_server.py``，持有 LIBERO 的
  MuJoCo env 和 EGL 渲染。它通过 RPC 传输对外暴露 ``reset``、``step``、
  ``chunk_step``、``render_camera(camera_name="agentview")``、``get_camera_meta``、
  ``cached_image`` 等方法，默认走 HTTP，加 ``--transport socket`` 则走 pickle
  分帧的 socket。
- **vla_server**，也就是 ``robots/libero/vla_server.py``，持有 Pi0.5 权重，
  通过同一套 RPC 传输（HTTP 或 socket）暴露 ``predict``。
- **sam3_server**，也就是 ``robots/libero/sam3_server.py``，持有 SAM 3.0，
  通过同一套 RPC 传输（HTTP 或 socket）支持文本或单个正点分割，只返回 top-1
  压缩 PNG mask。
- **Toolkit**，也就是 ``robots/libero/toolkit.py``，定义 LLM 能调的工具，比如
  ``pi0_pick``（交给 Pi0.5）、``move_to``、``rotate_wrist``、``back_project``、
  ``view_driver_state``、``finish`` 等。

Planner 能看到的工具
--------------------

LIBERO toolkit 默认暴露这些工具：

- ``pi0_pick(prompt)`` 调用 Pi0.5 生成一次抓取动作块，由 ``prompt`` 驱动，也就是
  一句自然语言的抓取指令。
- ``pi0_doubled(prompt)`` 调用 Pi0.5 生成一次非抓取的接触动作块，同样由
  ``prompt`` 驱动，比如拧旋钮、开关炉灶、短推。
- ``move_to(xyz)`` 是脚本化的 Cartesian 运动，移动到绝对世界坐标系下的目标点
  ``[x, y, z]``，单位是米，确定性执行，不走 VLA。
- ``move_pose(xyz)`` 也是脚本化的 Cartesian 运动，但会协同调整位置和腕部
  姿态（pitch 与 yaw），用于柜门前、低矮层架等姿态，避免解耦伺服卡死。
- ``rotate_wrist(target_yaw / delta_yaw)`` 是脚本化的腕关节旋转（绕世界 Z 轴），
  给一个绝对的 ``target_yaw`` 或相对的 ``delta_yaw``，单位是弧度。
- ``rotate_pitch(target_pitch / delta_pitch)`` 是脚本化的夹爪俯仰（绕世界 X 轴），
  给绝对的 ``target_pitch`` 或相对的 ``delta_pitch``，单位是弧度。
- ``release()`` 打开夹爪。
- ``set_gripper(gripper, steps)`` 保持当前姿态，驱动夹爪指令持续 ``steps`` 个
  env step，比如搬运途中收紧抓握。
- ``back_project(row, col)`` 把图像像素（``row`` 0 是顶部，``col`` 0 是左侧）
  反投影到世界坐标系下的 3D 点。
- ``segment(prompt=..., point=...)`` 用 SAM3 做分割：文本提示与单个
  ``[row, col]`` 正点二选一，把 top-1 mask 反投影得到 ``world_xyz``。mask 只在
  服务端与客户端之间使用，planner 只收到摘要和 artifact 路径。
- ``view_driver_state()`` 强制刷新一次状态 dump，包括图像、深度、camera meta 和
  ``states.json``。
- ``view_camera_meta(camera)`` 读取相机的标定元数据（``agentview`` 或 ``wrist``），
  用于定位。
- ``finish(status, summary)`` 结束当前 episode。``status`` 取 ``success``、
  ``failure`` 或 ``stuck``，``summary`` 是一段简短的自然语言总结，两者都必填。


每个工具跑完后都会重新渲染世界，所以下一轮 agent 上下文反映的是
动作后的状态。

Dashboard
---------

给 LIBERO 运行加上 ``--dashboard`` 打开本地监控页：

.. code-block:: bash

   rpent --env libero --dashboard \
     --suite libero_goal_task --task 1 --seed 0 --planner claude_code

Dashboard 会实时推送推理过程、agentview 与腕部相机、Pi0.5 叠加视图，以及动作
时间线。用 ``--dashboard-language zh-cn`` 可以切换到中文界面。

自带 VLA
--------

如果你有一个跟 LIBERO 兼容、但不是 Pi0.5 的 VLA，可以在不动 env 的情况下把
模型客户端换掉：

1. 写一个新的 ``vla_server.py``，暴露相同的 ``predict`` RPC 契约，http 或 socket
   都可以。
2. 用 ``--vla-endpoint [protocol://]host:port`` 指向它。
3. 如果暴露的工具要变，比如把 ``pi0_pick`` 改成 ``mymodel_pick``，相应更新
   ``robots/libero/toolkit.py``。

完整流程见 :doc:`../development/add_primitive`。
