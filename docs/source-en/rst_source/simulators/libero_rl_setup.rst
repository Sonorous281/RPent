LIBERO RL Installation and Resources
====================================

Prepare the sources, model service and simulator assets for :doc:`libero_rl`. Follow the online setup first; offline installation is an alternative at the end. RLinf training uses a separate `training setup <https://github.com/Sonorous281/RLinf/blob/dev/rpent-rlinf-integration/docs/source-en/rst_source/examples/agentic/rpent_libero_rl_setup.rst>`__ with different SGLang dependencies.

Install the Environments
------------------------

Use Linux x86_64, Python 3.12, Git and `uv <https://docs.astral.sh/uv/>`__. The simulator needs NVIDIA EGL libraries; the model server needs a working driver compatible with its CUDA 13 wheels. Install the NVIDIA driver through your host/platform administrator. On Ubuntu, the remaining system packages are:

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

Both LIBERO workflows default to ``MUJOCO_GL=egl`` and preserve an explicitly set backend such as ``osmesa`` for CPU rendering.

This recipe targets the published ``Sonorous281/RPent:dev/rpent-rlinf-integration`` and ``Sonorous281/RLinf:dev/rpent-rlinf-integration`` feature branches. Record both source revisions with each run. Run the requirements command from the checkout root. It installs RPent in editable mode, the supported RLinf runtime, SAM3 and the LIBERO-PRO simulator; this workflow does not need OpenPi/Pi0.5 or an action-model checkpoint. The requirements use the paired sibling RLinf checkout and the public RPent SAM3/LIBERO forks. They do not select the older ``rlinf`` extra pinned to ``release/v0.4``. Record their resolved revisions and ``uv pip freeze`` output with an experiment.

Create the model-serving environment in a **new terminal**, from the same checkout. SGLang 0.5.23 pins its own Torch, Transformers and CUDA dependencies:

.. code-block:: bash

   uv venv --python 3.12 .venv-sglang
   source .venv-sglang/bin/activate
   uv pip install --prerelease=allow "sglang==0.5.23" "torchvision==0.29.2"

Do not inherit ``LD_LIBRARY_PATH`` or ``PYTHONPATH`` from an RL/SFT environment when opening the serving terminal. Activating a venv changes Python/PATH; it does not undo another environment's CUDA library overrides. An existing site profile may be used explicitly in place of standard venv activation when that host needs its own CUDA/EGL driver paths; none is sourced by these launchers.

.. list-table::
   :header-rows: 1

   * - Component
     - RPent / simulator
     - Model serving
   * - Python
     - 3.12
     - 3.12
   * - Torch / CUDA
     - 2.11.0 / 12.8
     - 2.14.1 / 13.0
   * - Transformers
     - 4.57.6
     - 5.19.0
   * - Policy server
     - HTTP client
     - SGLang 0.5.23, kernel 0.4.10
   * - Planner
     - pydantic-ai-slim 2.53.0, harness 0.53.0
     - Qwen tool/reasoning parsers
   * - Simulator
     - MuJoCo 3.3.0, robosuite 1.5.2, NumPy 1.26.4
     - —
   * - Segmentation
     - SAM3 0.1.0, torchvision 0.26.0
     - —

These are the compatible stack versions used for the existing rollout validation. ``requirements/libero-rl.txt`` fixes the main simulator versions; it is not a complete transitive lockfile. The serving wheels require their matching CUDA driver even when an older CUDA toolkit is installed on the host.

Prepare Models and Simulator Assets
-----------------------------------

In the RPent environment, source ``env.sh`` before downloads and every rollout session. To choose shared storage, export ``RPENT_DATA_ROOT=/your/data/root`` **before** sourcing it. By default it uses ``$XDG_DATA_HOME/rpent``, or ``$HOME/.local/share/rpent`` when ``XDG_DATA_HOME`` is unset. Explicit ``SAM3_CHECKPOINT_PATH``, ``LIBERO_CONFIG_PATH`` and ``LIBERO_PRO_ASSET_PATH`` take precedence over these defaults.

Download the complete Qwen directory, including its tokenizer, processor and chat template. SAM3 access must be granted on its model page before login:

.. code-block:: bash

   source .venv/bin/activate
   source tools/libero_rl_eval/env.sh
   unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE
   hf download Qwen/Qwen3.8-27B --local-dir "$RPENT_DATA_ROOT/models/Qwen3.8-27B"
   hf auth login
   hf download facebook/sam3 sam3.pt --local-dir "$(dirname "$SAM3_CHECKPOINT_PATH")"
   mkdir -p "$LIBERO_CONFIG_PATH" "$LIBERO_PRO_ASSET_PATH"
   env -u LIBERO_PRO_ASSET_PATH liberopro-download-assets --assets-dir "$LIBERO_PRO_ASSET_PATH" --skip-existing --no-cache

For a network that uses the HF mirror, set ``HF_ENDPOINT=https://hf-mirror.com`` before these commands. Clear proxy variables only if your network requires direct access. ``hf download`` replaces any dependency on a private ``hfd.sh``. The downloader's ``LIBERO_PRO_ASSET_PATH`` fallback selects an existing asset tree, so the download command temporarily removes that variable.

BDDL files and initialization states ship with ``rpent-liberopro``. Generate the configuration using that installed package, while keeping the downloaded assets in your data directory:

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

Check that the printed assets, BDDL and initialization-state directories exist. Keep the ``benchmark_root`` at the installed package path: RLinf checks it when loading a simulator. No demonstration dataset download is needed for these rollouts.

Offline Installation
--------------------

Prepare separate **complete** wheelhouses for the two environments on a networked Linux x86_64/Python 3.12 host. First follow the online installation above. In each environment, build its requirements into its own wheelhouse:

.. code-block:: bash

   source .venv/bin/activate
   uv pip install pip wheel
   python -m pip wheel --pre --wheel-dir wheelhouse-rpent --extra-index-url https://download.pytorch.org/whl/cu128 -r requirements/libero-rl.txt setuptools wheel
   source .venv-sglang/bin/activate
   uv pip install pip wheel
   python -m pip wheel --pre --wheel-dir wheelhouse-sglang "sglang==0.5.23" "torchvision==0.29.2"

Transfer the checkout, both wheelhouses, model directories and simulator assets to the offline host. Match OS, CPU architecture, Python ABI and CUDA drivers; extensions without a prebuilt wheel also need their build toolchain on the preparation host. A few cached wheels are insufficient to reconstruct a full environment. On the offline host, create fresh venvs and install the complete wheel set for each environment. ``--no-deps`` prevents Git URL metadata from contacting the network; all dependency wheels must already be present.

.. code-block:: bash

   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install --no-index --no-deps wheelhouse-rpent/*.whl
   uv pip install --no-index --no-deps --no-build-isolation -e .
   uv venv --python 3.12 .venv-sglang
   source .venv-sglang/bin/activate
   uv pip install --no-index --no-deps wheelhouse-sglang/*.whl

The offline Python interpreter must be installed locally before ``uv venv``. Run ``uv pip check`` in each environment. Re-run the configuration-generation step after relocating the simulator, since its YAML contains absolute paths. With all assets present, export ``HF_HUB_OFFLINE=1`` and ``TRANSFORMERS_OFFLINE=1`` in runtime terminals. Keep environment installation separate from active training processes.

Return to :doc:`libero_rl` to start the services and run your first task.
