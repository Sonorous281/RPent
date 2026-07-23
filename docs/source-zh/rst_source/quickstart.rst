快速开始
========

本页把 ``README.md`` 里的 Quick Start 搬进了文档，假设你已经读完
:doc:`installation`，克隆好了 RPent 并执行过 ``pip install -e ".[full]"``。

1. 配置 API key 与 checkpoint
------------------------------

导出 Anthropic 密钥和 VLA 检查点的路径：

.. code-block:: bash

   # Anthropic 密钥; 使用官方端点时无需 export base url。
   export ANTHROPIC_BASE_URL=https://xxx
   export ANTHROPIC_API_KEY=sk-xxx

   # VLA 检查点, 从下面地址下载:
   # https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT
   export PI05_CHECKPOINT_PATH=/path/to/rlinf-pi05-libero-130-fullshot-sft
   export LIBERO_TYPE=pro
   export CUDA_VISIBLE_DEVICES=0

2. 跑一个 LIBERO 任务
---------------------

用 ``claude_code`` planner 跑一个 LIBERO PRO 任务，这里选套件
``libero_object_swap``、任务 ``2``、种子 ``0``：

.. code-block:: bash

   rpent --env libero --suite libero_object_swap --task 2 --seed 0 \
     --planner claude_code --model claude-opus-4-8

其他 planner（``api`` 和 ``codex``）以及各家模型提供方的配置，见
:doc:`usage/configure_planner`。

3. 用 dashboard 观察运行
------------------------

加上 ``--dashboard`` 会打开浏览器里的监控页面。它先展示一个启动页让你确认
配置，随后开始实时推送：agent 的推理过程、实时相机与 Pi0 视图、动作时间线，
以及回放片段。再加上 ``--dashboard-language zh-cn`` 就切换到中文界面。

.. code-block:: bash

   rpent --env libero --dashboard --dashboard-language zh-cn \
     --suite libero_goal_task --task 1 --seed 0 --planner claude_code

关键 CLI 选项
-------------

``rpent`` 日常最常用的几个 flag:

.. list-table::
   :header-rows: 1
   :widths: 22 15 63

   * - Flag
     - 默认值
     - 说明
   * - ``--env``
     - 必填
     - 环境后端。当前支持 ``libero``。
   * - ``--suite``
     - 必填
     - 任务套件，如 ``libero_object_task``、``libero_spatial_swap``
   * - ``--task``
     - 必填
     - 套件内的任务 id
   * - ``--seed``
     - ``0``
     - 随机种子
   * - ``--planner``
     - ``api``
     - 决策大脑，可选 ``api`` | ``claude_code`` | ``codex``
   * - ``--model``
     - —
     - 模型 id; ``api`` 下要带 provider 前缀 (``anthropic:…``,
       ``openai:…``, ``openai-chat:…``)
   * - ``--max-turns``
     - ``100``
     - Agent 最大轮数
   * - ``--max-tokens``
     - ``8192``
     - LLM 每次回复的最大 token 数
   * - ``--no-images``
     - 关
     - 纯文本模式：不向模型发送图片字节 (用于不支持图片输入的模型)
   * - ``--max-episode-steps``
     - ``10000``
     - Env 最大 step 数
   * - ``--libero-type``
     - ``LIBERO_TYPE`` 或 ``pro``
     - LIBERO 变体：``standard`` | ``pro`` | ``plus``
   * - ``--cuda-device``
     - 继承
     - 暴露给 env / vla server 的 GPU 设备
   * - ``--dashboard``
     - 关
     - 为本次运行启动本地 dashboard
   * - ``--dashboard-language``
     - ``en``
     - Dashboard UI 语言：``en`` | ``zh-cn``
   * - ``--env-endpoint``
     - —(自动 spawn)
     - 已在运行的 env_server 的 ``[protocol://]host:port``
       (``protocol=http|socket``，默认 ``http``). 留空则本地起一个。
   * - ``--vla-endpoint``
     - —(自动 spawn)
     - 已在运行的 vla_server 的 ``[protocol://]host:port`` (同上).
       留空则本地起一个。

跑起来后应该看到什么
--------------------

一次成功的运行是这样的：

1. env_server 和 vla_server 起来后，各打印一行
   ``RPC server listening on http://127.0.0.1:<port>``。
2. 每一轮 agent 的推理都会输出到终端，或推送到 dashboard。
3. 当 LLM 调用 ``finish(status, summary)`` 且 ``status="success"``
   (或 ``"failure"`` / ``"stuck"``) 时结束；触达
   ``--max-turns`` / ``--max-episode-steps`` 上限时也会结束。
4. 写出两份产物: ``<output_dir>/transcript_*.json`` 记录逐轮的完整对话，
   ``<output_dir>/episode.mp4`` 是渲染出的运行录像。

万一出了问题，可以查 :doc:`installation` 页底部提到的三份日志文件。
