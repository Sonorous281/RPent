安装
====

RPent 用一条 ``pip install`` 即可安装。optional-dependency extra 会从
PyPI 拉取已发布的 openpi 与 LIBERO 仿真器包。

先决条件
--------

- Linux + NVIDIA GPU (LIBERO 通过 EGL 渲染)。
- 与显卡匹配的 CUDA 12.x 驱动。
- Python 3.10–3.12。
- ``git``、``bash``、以及能编译 MuJoCo / robosuite 的 C 工具链。

同时你还需要准备两样东西。

- 至少一个 LLM 提供方的 API key，用于驱动推理大脑，可以是 Anthropic、OpenAI，
  或任何 OpenAI 兼容的 chat 接口。
- 一个 VLA 检查点。LIBERO 上的 Pi0.5 推荐用
  `HuggingFace: RLinf-Pi05-LIBERO-130-fullshot-SFT
  <https://huggingface.co/RLinf/RLinf-Pi05-LIBERO-130-fullshot-SFT>`_。

1. 用 pip 安装 RPent
--------------------

克隆 RPent (它提供 CLI 和运行配置)，再按需选择 extra 安装：

.. code-block:: bash

   git clone https://github.com/RLinf/RPent rpent && cd rpent
   pip install -e ".[full]"

``.[full]`` 是默认的端到端组合，包含 openpi 的 Pi0.5 VLA 和 LIBERO-PRO 仿真器，
跑在 RLinf 运行时之上。

可用的 extra:

.. list-table::
   :header-rows: 1

   * - Extra
     - 安装内容
   * - ``.[full]``
     - ``rlinf`` + ``openpi`` + ``libero-pro``, 默认运行组合
   * - ``.[libero-pro]``
     - 仅基础 LIBERO + LIBERO-PRO 仿真器
   * - ``.[libero-plus]``
     - 基础 LIBERO + LIBERO-plus 仿真器
   * - ``.[libero]``
     - 仅基础 LIBERO
   * - ``.[openpi]``
     - 仅 openpi VLA
   * - ``.[rlinf]``
     - 仅 RLinf 运行时

2. 下载仿真资产
---------------

PyPI wheel 不包含大体积仿真资产。安装后需一次性下载：

.. code-block:: bash

   libero-download-assets --skip-existing      # 基础 LIBERO
   liberopro-download-assets --skip-existing   # LIBERO-PRO —— .[libero-pro] / .[full]
   liberoplus-download-assets --skip-existing  # LIBERO-plus —— .[libero-plus]

.. tip::

   访问 Hugging Face 较慢时，可通过 ``HF_ENDPOINT`` 走镜像加速下载：

   .. code-block:: bash

      HF_ENDPOINT=https://hf-mirror.com liberopro-download-assets --skip-existing

3. (可选) 真实机器人依赖
------------------------

Franka 和 SO-101 的支持正在逐步接入。每个机器人的驱动都会以一个包的形式放在
``robots/<name>/`` 下，并附带 ``README.md`` 说明它对 SDK 和固件的要求。当前进度
参见 :doc:`usage/franka` 和 :doc:`usage/so101`。

验证安装
--------

最快的验证办法是端到端跑通一个 LIBERO 任务，具体步骤见 :doc:`quickstart`。
只要跑通，就说明 env server、VLA server 和推理大脑三者都正常。

万一出错，可以按下面三份日志排查:

- env server 的标准输出和标准错误都写到 ``<output_dir>/env_server.log``。
- VLA server 的日志在 ``<output_dir>/vla_server.log``。
- agent 本身的运行日志在 ``<output_dir>/run.log``。

三份日志都放在本次运行的临时目录下，所以每一次失败的运行都是自包含的，排查
起来很方便。
