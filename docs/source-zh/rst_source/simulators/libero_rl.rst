LIBERO RL 轨迹采集
==================

``libero_rl`` 使用语言模型规划器调用六种机器人工具，在 LIBERO-PRO 中执行任务，不依赖动作模型、任务指南或任务记忆。独立 RPent 采集与 RLinf 训练使用相同的机器人接口。本页介绍采集，训练配置和启动入口由 RLinf 仓库维护。

概览
----

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 动作模型

      不使用 VLA；工具直接控制末端执行器。

   .. grid-item-card:: 规划器

      支持图像和工具调用的 Qwen；示例使用 Qwen3.8-27B。

   .. grid-item-card:: 任务

      LIBERO-PRO 的四个 suite，共 40 个任务。

   .. grid-item-card:: 硬件

      Linux NVIDIA GPU；示例分配两卡模型服务、一卡 SAM3 和一卡仿真。

任务
~~~~

.. list-table::
   :header-rows: 1

   * - 范围
     - 内容
   * - LIBERO-PRO
     - ``libero_object``、``libero_goal``、``libero_spatial``、``libero_10``，各含 task 0–9。
   * - Seed
     - ``--seed`` 选择任务初始状态；批量采集可覆盖多个 seed。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - 项目
     - 行为
   * - 观测
     - 仅支持 adaptive：初始与动作后返回实测状态，``view_env`` 返回两个相机图像。
   * - 动作
     - ``move_to`` / ``set_gripper`` / ``view_env`` / ``segment`` / ``back_project`` / ``finish``
   * - 奖励与成功判定
     - 以最后状态中的 LIBERO ``terminated`` 标志为准，规划器的 ``finish`` 不代表成功。
   * - 任务指令
     - 任务描述与工具提示词；默认 ``simple``，可选简短推理或几何信息变体。

每次任务运行启动独立仿真环境。规划器调用 ``finish``、耗尽预算或超时后结束，多个任务可共用 SAM3。资源分配示例不代表最低显存要求，应先验证单任务再增加并发。

安装与资源准备
--------------

先完成 :doc:`libero_rl_setup`，使用配套的 ``Sonorous281/RPent`` 与 ``Sonorous281/RLinf`` integration 分支，准备仿真和模型服务两个环境、完整 Qwen 模型、SAM3 权重及 LIBERO-PRO 资源。该流程不需要 Pi0.5 权重。

后续命令均在 RPent 仓库根目录执行。三个终端使用同一源码和资源路径；需要自定义存储时，在各终端导出 ``RPENT_DATA_ROOT`` 后再加载 ``env.sh``。

运行一个任务
------------

在新终端激活 SGLang 并启动模型服务。``serve.sh`` 使用当前解释器（或 ``RPENT_SGLANG_PYTHON``），必须显式设置 ``MODEL_PATH``。默认服务名称为 Qwen3.8-27B，使用 GPU 2–3、TP2、MTP 和端口 30080。可以通过 ``SERVE_GPUS``、``SERVE_PORT``、``MODEL_PATH``、``MODEL_NAME``、``SERVE_MEM_FRACTION`` 和 ``SERVE_CONCURRENCY`` 覆盖服务设置。没有 MTP 权重的模型使用 ``ENABLE_MTP=0``。

.. code-block:: bash

   source .venv-sglang/bin/activate
   MODEL_PATH="$HOME/.local/share/rpent/models/Qwen3.8-27B" SERVE_GPUS=2,3 \
     bash tools/libero_rl_eval/serve.sh

若使用 ``XDG_DATA_HOME`` 或自定义存储，将 ``MODEL_PATH`` 改为下载目录。本终端只使用 SGLang 环境；``env.sh`` 要求 RPent 解释器，仅在仿真和 SAM3 终端加载。

``curl http://127.0.0.1:30080/v1/models`` 列出模型时，表示服务已经就绪。在第二个终端激活 RPent 环境，启动共享 SAM3：

.. code-block:: bash

   source .venv/bin/activate
   source tools/libero_rl_eval/env.sh
   python -m rpent.robots.components.sam3_server --cuda-device 0 --port 8112

在第三个终端激活 RPent，在 GPU 1 上运行单条轨迹：

.. code-block:: bash

   source .venv/bin/activate
   source tools/libero_rl_eval/env.sh
   export RPENT_MODEL=openai-chat:qwen3.8-27b
   export RPENT_BASE_URL=http://127.0.0.1:30080/v1
   export RPENT_SAM3_ENDPOINT=http://127.0.0.1:8112
   bash tools/libero_rl_eval/run-libero-rl.sh \
     --suite libero_object --task 0 --seed 0 --cuda-device 1

``--cuda-device`` 指定仿真 GPU，SAM3 endpoint 则独立指定共享服务。不设置 endpoint 时，每条轨迹会在仿真 GPU 上启动自己的 SAM3 服务。额外的命令行参数会传给 ``rpent``。

查看结果
--------

默认输出目录为 RPent 仓库下的 ``logs/libero-rl-adaptive-<prompt>-<time>``，可以通过 ``RPENT_OUTPUT_DIR`` 修改。每条轨迹包含：

* ``result.json``：native success、错误、轮数和 token 用量；
* ``states.json``：测量状态和原生 ``terminated`` 标志；
* ``transcript_*.json``：包含 thinking 和工具调用的原生消息；
* ``rollout.json``、``episode.mp4`` 及相机和状态文件；
* ``trajectories/<task_id>/``：任务结果、规划器请求与响应、工具调用、已执行动作、观测及附件。

启动器默认开启 RPent 共用的轨迹记录器。规划器记录包含历史压缩后的 Pydantic AI 输入及对应响应，记录的是 SDK 调用，不是原始 HTTP 报文。任务元数据和关联记录使用 schema version 5。批量入口发布轨迹前，会检查任务正常结束、每轮响应完整可用、附件可读，以及记录的成功状态与原生环境一致。历史 ``api_trace/`` 仍可读取，并沿用原来的报文与摘要校验。

批量采集任务和 Seed
-------------------

``jobs/panel40.txt`` 包含 ``libero_object``、``libero_goal``、``libero_spatial`` 和 ``libero_10`` 各自的 task 0–9，默认 adaptive 模式、seed 0。每行为 ``adaptive suite task seed``，``--seeds`` 可以用逗号分隔的列表或闭区间覆盖清单中的 seed。

.. code-block:: bash

   python tools/libero_rl_eval/run_batch.py run \
     "$RPENT_DATA_ROOT/rollouts/new-run" \
     --jobs tools/libero_rl_eval/jobs/panel40.txt \
     --seeds 3-66 --gpus 1 --per-gpu 4

模型和共享 SAM3 使用单条轨迹示例中的环境变量。``--endpoints URL,URL`` 可以将 worker 分配给其他本地模型服务。``--start-interval`` 控制初次启动 worker 的间隔；设置 ``--per-gpu`` 时须考虑仿真 GPU 的可用显存和初始化负载。

.. list-table::
   :header-rows: 1

   * - Batch 参数
     - 默认值
     - 含义
   * - ``--jobs``
     - 必填
     - 四字段任务清单
   * - ``--seeds``
     - 使用清单的 seed
     - 逗号分隔的 seed 或区间，例如 ``0,3-66``
   * - ``--gpus``、``--per-gpu``
     - GPU 必填，每 GPU 1 个 worker
     - 环境 GPU 和每卡 worker 数
   * - ``--prompt``
     - ``simple``
     - 整批使用一个 prompt 变体
   * - ``--endpoints``
     - ``RPENT_BASE_URL``
     - 逗号分隔的策略 ``/v1`` endpoint
   * - ``--start-interval``
     - ``1`` 秒
     - 初次启动 worker 的间隔
   * - ``--episode-timeout``
     - ``14400`` 秒
     - 单条轨迹的外层进程超时
   * - ``--infrastructure-retries``
     - ``1``
     - 每次启动中，错误或审计失败后的额外尝试数

每个任务独立保存各次 ``attempt-NNN``。正常任务失败也是有效数据。基础设施异常和审计失败会保留 ``INVALID.json``，默认重试一次；中断的尝试保存 ``INTERRUPTED.json``。只有通过审计的轨迹才会获得 ``COMPLETED.json``，并写入原子更新的 ``valid-index.json``，暂停采集时仍可使用已完成的数据。SIGINT/SIGTERM 会终止本批次的轨迹进程组，重复相同命令即可恢复。目录锁防止两个采集进程重复启动同一队列；恢复时 manifest 会拒绝任务、模型、prompt 或生成参数变化。

.. code-block:: bash

   python tools/libero_rl_eval/run_batch.py summary \
     "$RPENT_DATA_ROOT/rollouts/new-run"

summary 会重新审计已封存的轨迹，报告有效数量和原生成功数量。采集轨迹的 SFT 数据准备与训练见配套 ms-swift checkout 的 ``ms-swift/examples/rpent``。

模型服务配置
------------

模型服务仅监听 ``127.0.0.1``。下表参数控制服务；``serve.sh`` 后的命令行参数直接传给 SGLang。

.. list-table::
   :header-rows: 1

   * - 服务变量
     - 默认值
     - 含义
   * - ``RPENT_SGLANG_PYTHON``
     - ``python``
     - 已激活 SGLang 环境的解释器；不负责切换 CUDA 库
   * - ``SERVE_GPUS``
     - ``2,3``
     - 可见物理 GPU，数量决定 TP
   * - ``MODEL_PATH``
     - 必填
     - 本地权重目录
   * - ``MODEL_NAME``
     - ``qwen3.8-27b``
     - 服务模型名称，须与 ``RPENT_MODEL`` 的 ``openai-chat:`` 后缀一致
   * - ``SERVE_PORT``
     - ``30080``
     - 本地模型端口
   * - ``SERVE_MEM_FRACTION``
     - ``0.8``
     - SGLang 静态显存比例
   * - ``SERVE_CONCURRENCY``
     - ``8``
     - 最大同时运行的模型请求数
   * - ``ENABLE_MTP``
     - ``1``
     - EAGLE/MTP：3 steps、top-k 1、4 draft tokens；没有 MTP 权重的模型设为 0

规划器配置
----------

``planner.sh`` 使用 Qwen thinking 的采样参数：temperature 1.0、top_p 0.95、top_k 20。Qwen3.8 默认使用 ``reasoning_effort=xhigh``；其他 Qwen 模型不发送 effort 等级，但仍启用 thinking。模型服务使用 ``qwen3_coder`` 解析工具调用，使用 ``qwen3`` 解析推理。

.. list-table:: 可覆盖的参数
   :header-rows: 1
   :widths: 45 55

   * - 变量
     - 默认值
   * - ``RPENT_PROMPT``
     - ``simple``
   * - ``RPENT_MAX_TURNS`` / ``RPENT_MAX_EPISODE_STEPS``
     - 100 / 10000
   * - ``RPENT_MAX_TOKENS`` / ``RPENT_CONTEXT_WINDOW``
     - 32768 / 262144
   * - ``RPENT_COMPACTION_FRACTION``
     - 0.7
   * - ``RPENT_REASONING_EFFORT``
     - Qwen3.8 为 ``xhigh``，其他模型为 ``none``
   * - ``RPENT_PLANNER_TIMEOUT_S``
     - 7200
   * - ``RPENT_PRESENCE_PENALTY`` / ``RPENT_REPETITION_PENALTY``
     - 未设置，可显式覆盖采样参数

复现 SFT checkpoint 的采样惩罚时，显式设置 ``RPENT_PRESENCE_PENALTY=1.5`` 和 ``RPENT_REPETITION_PENALTY=1.0``。``--observation-mode adaptive`` 可显式传入，image/text 模式不受支持。安装检查可调低 ``RPENT_MAX_TURNS``；正式采集保留既定预算。

历史 thinking 通过 ``reasoning_content`` 保留，并由模型自身的 chat template 渲染。RPent 用 ``<tool_response>`` 标签包裹工具图片的 user 消息，让 Qwen3.5 保留此前的 thinking；Qwen3.8 还会设置 ``preserve_thinking=true``。上下文管理会主动改变可见历史，最终压缩后的 SDK 请求保存在轨迹中。达到输出 token 上限的回答属于不完整训练输出，无法通过轨迹审计。

Prompt 定义位于 ``robots/libero_rl/prompt_bundle.py``：``simple`` 包含工具契约；``simple_brief`` 另外要求每次决策尽量不超过 256 个推理 token，这是提示词指导，不是解码器硬上限；``base`` 增加机器人坐标系和执行方法；``base_gripper`` 增加手指几何；``base_gripper_camera`` 进一步增加相机几何。

RLinf 的消息窗口压缩会将估算的图像 token 计入触发条件，不把图像字节当作文本计数。消息截断仍采用 Harness 的滑动窗口并保留初始任务。这个估算用于上下文管理，不代表模型实际用量。按 token 数保留历史的配置仍使用 Harness 原有行为。

RLinf 训练
----------

训练入口见配套分支的 `RLinf 多模态训练说明 <https://github.com/Sonorous281/RLinf/blob/dev/rpent-rlinf-integration/docs/source-zh/rst_source/examples/agentic/rpent_libero_rl.rst>`__，其中包含安装、配置、启动、恢复和评测。RPent 负责仿真与工具；RLinf 负责模型生成、Actor/Critic 更新与权重同步。``tools.rpent.python`` 必须指向 RPent 解释器，子进程还需正确的 CUDA/EGL 环境。

本页采集默认为 LIBERO-PRO，RLinf 多模态配方默认标准 LIBERO；采集环境变量不会覆盖 RLinf 配置。比较结果前，对齐基准、任务与 seed、模型、``agentloop.prompt``、``agentloop.chat_template_kwargs``、``agentloop.history_compaction``、``algorithm.sampling_params`` 和 ``runner.seq_length``。

SFT 数据准备见配套 ms-swift checkout 的 ``ms-swift/examples/rpent/README.md``。导出器必须支持 schema version 5，并保留压缩后的请求、图像内容与顺序、历史 thinking 及 assistant loss mask。

常见问题
--------

.. list-table::
   :header-rows: 1

   * - 现象
     - 检查
   * - 无法访问模型服务
     - 确认 ``/v1/models`` 返回配置的模型名，检查 SGLang 终端及其独立环境。
   * - 仿真初始化失败
     - 确认 EGL/驱动可用，``config.yaml`` 中 assets、BDDL 与初始状态目录存在。
   * - 轨迹被标为无效
     - 读取 ``INVALID.json`` 和该次尝试的日志。正常任务失败可保留；响应截断、缺失附件或基础设施异常需修复后重试。

.. toctree::
   :hidden:

   libero_rl_setup
