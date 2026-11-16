LIBERO RL 安装与资源准备
========================

为 :doc:`libero_rl` 准备源码、模型服务和仿真资源。首次部署按下方顺序执行；离线安装放在最后。RLinf 训练使用独立的 `训练安装说明 <https://github.com/Sonorous281/RLinf/blob/dev/rpent-rlinf-integration/docs/source-zh/rst_source/examples/agentic/rpent_libero_rl_setup.rst>`__，不要混装两套 SGLang 环境。

安装环境
--------

使用 Linux x86_64、Python 3.12、Git 和 `uv <https://docs.astral.sh/uv/>`__。仿真器需要 NVIDIA EGL 库；模型服务需要支持其 CUDA 13 wheel 的驱动。 NVIDIA 驱动由机器或平台管理员安装。Ubuntu 的其余系统依赖如下：

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install -y git build-essential ffmpeg libegl1 libopengl0 libgl1 libglib2.0-0
   git clone --branch dev/rpent-rlinf-integration https://github.com/Sonorous281/RLinf.git RLinf
   git clone --branch dev/rpent-rlinf-integration https://github.com/Sonorous281/RPent.git RPent
   cd RPent
   git rev-parse HEAD
   git -C ../RLinf rev-parse HEAD
   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install --prerelease=allow --torch-backend cu128 -r requirements/libero-rl.txt
   source tools/libero_rl_eval/env.sh

两种 LIBERO 工作流默认使用 ``MUJOCO_GL=egl``；若显式设置 ``osmesa`` 等后端，会保留该设置，供 CPU 渲染使用。

本示例使用已发布的 ``Sonorous281/RPent:dev/rpent-rlinf-integration`` 和 ``Sonorous281/RLinf:dev/rpent-rlinf-integration`` 功能分支。每次运行应记录两个仓库的提交。依赖安装命令须在仓库根目录执行。它以 editable 方式安装 RPent，并安装支持的 RLinf 运行时、SAM3 和 LIBERO-PRO 仿真器；本流程不需要 OpenPi/Pi0.5 或动作模型权重。依赖文件使用同级目录中的 RLinf checkout 及公开的 RPent SAM3/LIBERO forks，不选择固定到旧 ``release/v0.4`` 的 ``rlinf`` extra。实验中应记录这些依赖实际解析到的提交，以及 ``uv pip freeze`` 输出。

另开终端，进入同一仓库，再创建模型服务环境。SGLang 0.5.23 自身锁定所需 Torch、Transformers 和 CUDA 依赖：

.. code-block:: bash

   uv venv --python 3.12 .venv-sglang
   source .venv-sglang/bin/activate
   uv pip install --prerelease=allow "sglang==0.5.23" "torchvision==0.29.2"

模型服务终端不要继承 RL/SFT 环境的 ``LD_LIBRARY_PATH`` 或 ``PYTHONPATH``。激活 venv 只切换 Python/PATH，不会撤销其他环境的 CUDA 库覆盖。机器确实需要专用 CUDA/EGL 驱动路径时，可以显式使用已有站点 profile 代替标准 venv 激活；这些启动脚本不会自动加载站点 profile。

.. list-table::
   :header-rows: 1

   * - 组件
     - RPent / 仿真环境
     - 模型服务环境
   * - Python
     - 3.12
     - 3.12
   * - Torch / CUDA
     - 2.11.0 / 12.8
     - 2.14.1 / 13.0
   * - Transformers
     - 4.57.6
     - 5.19.0
   * - 策略服务
     - HTTP 客户端
     - SGLang 0.5.23、kernel 0.4.10
   * - 规划器
     - pydantic-ai-slim 2.53.0、harness 0.53.0
     - Qwen 工具和推理解析器
   * - 仿真器
     - MuJoCo 3.3.0、robosuite 1.5.2、NumPy 1.26.4
     - —
   * - 分割
     - SAM3 0.1.0、torchvision 0.26.0
     - —

这些兼容版本用于已有 rollout 验证。``requirements/libero-rl.txt`` 固定主要仿真依赖版本，并非完整的传递依赖锁文件。即使机器装有较旧 CUDA toolkit，服务端 wheel 仍需要兼容其版本的 CUDA 驱动。

准备模型与仿真资源
------------------

在 RPent 环境中，下载和每次运行前都 source ``env.sh``。如需共享存储，先设置 ``RPENT_DATA_ROOT=/your/data/root``，再 source。默认数据目录为 ``$XDG_DATA_HOME/rpent``；没有设置 ``XDG_DATA_HOME`` 时使用 ``$HOME/.local/share/rpent``。显式设置的 ``SAM3_CHECKPOINT_PATH``、 ``LIBERO_CONFIG_PATH`` 和 ``LIBERO_PRO_ASSET_PATH`` 优先于这些默认值。

下载完整 Qwen 目录，包括 tokenizer、processor 和 chat template。 SAM3 须先在模型页面获得访问权限，再登录下载：

.. code-block:: bash

   source .venv/bin/activate
   source tools/libero_rl_eval/env.sh
   unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
   hf download Qwen/Qwen3.8-27B --local-dir "$RPENT_DATA_ROOT/models/Qwen3.8-27B"
   hf auth login
   hf download facebook/sam3 sam3.pt --local-dir "$(dirname "$SAM3_CHECKPOINT_PATH")"
   mkdir -p "$LIBERO_CONFIG_PATH" "$LIBERO_PRO_ASSET_PATH"
   env -u LIBERO_PRO_ASSET_PATH liberopro-download-assets --assets-dir "$LIBERO_PRO_ASSET_PATH" --skip-existing --no-cache

需要 HF 镜像时，在命令前设置 ``HF_ENDPOINT=https://hf-mirror.com``。只有网络要求直连时才清除代理变量。``hf download`` 不依赖私有 ``hfd.sh``。下载器会将 ``LIBERO_PRO_ASSET_PATH`` 解释为已有资源树，因此下载命令临时移除该变量，目标目录仍通过参数显式传入。

BDDL 和初始状态文件由 ``rpent-liberopro`` 安装包提供。根据实际安装位置生成配置，同时将下载的大资源保留在自己的数据目录中：

.. code-block:: bash

   python - <<'PYCONFIG'
   import os
   from pathlib import Path
   import yaml
   import liberopro.liberopro as libero
   config = libero.get_default_path_dict()
   config["assets"] = str(Path(os.environ["LIBERO_PRO_ASSET_PATH"]).resolve())
   config["datasets"] = str(Path(os.environ["RPENT_DATA_ROOT"]) / "libero-pro/datasets")
   path = Path(os.environ["LIBERO_CONFIG_PATH"]) / "config.yaml"
   path.write_text(yaml.safe_dump(config))
   print(path.read_text())
   PYCONFIG

确认打印出的 assets、BDDL 和初始状态目录均存在。``benchmark_root`` 应保留为实际安装的包路径，RLinf 会在加载仿真器时检查它。此类 rollout 不需要下载 demonstration 数据集。

离线安装
--------

在可联网的 Linux x86_64 / Python 3.12 机器上，为两个环境分别准备“完整” wheelhouse。先按前文完成在线安装，然后在各自环境中构建对应依赖：

.. code-block:: bash

   source .venv/bin/activate
   uv pip install pip wheel
   python -m pip wheel --pre --wheel-dir wheelhouse-rpent --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements/libero-rl.txt setuptools wheel
   source .venv-sglang/bin/activate
   uv pip install pip wheel
   python -m pip wheel --pre --wheel-dir wheelhouse-sglang "sglang==0.5.23" "torchvision==0.29.2"

将仓库、两套 wheelhouse、模型目录和仿真资源传到离线机器。操作系统、CPU 架构、Python ABI 和 CUDA 驱动须匹配；没有预编译 wheel 的扩展还需要在准备机器上安装构建工具链。少量缓存 wheel 不能重建完整环境。离线机器上分别创建新 venv 并安装各自完整依赖；``--no-deps`` 避免包内 Git URL 元数据触发联网，因此所有依赖 wheel 都必须已经准备好。

.. code-block:: bash

   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install --no-index --no-deps wheelhouse-rpent/*.whl
   uv pip install --no-index --no-deps --no-build-isolation -e .
   uv venv --python 3.12 .venv-sglang
   source .venv-sglang/bin/activate
   uv pip install --no-index --no-deps wheelhouse-sglang/*.whl

离线执行 ``uv venv`` 前，机器上须已安装对应 Python 解释器。在两个环境中分别运行 ``uv pip check``。迁移仿真器后重新执行配置生成步骤，因为 YAML 中保存的是绝对路径。全部资源就绪后，可在运行终端设置 ``HF_HUB_OFFLINE=1`` 和 ``TRANSFORMERS_OFFLINE=1``。环境安装和维护须与正在运行的训练进程分开。

资源准备完毕后，返回 :doc:`libero_rl`，启动服务并运行第一个任务。
