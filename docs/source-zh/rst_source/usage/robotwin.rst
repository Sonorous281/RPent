RoboTwin
========

RPent 复用 RLinf 的 ``RoboTwinEnv``，并由它唯一持有 RoboTwin native task。
RPent 进程只通过薄 RPC bridge 访问环境：

.. code-block:: text

   RPent Toolkit -> RPent EnvServer -> RLinf RoboTwinEnv
   -> RoboTwin VectorEnv/SubEnv -> native task

依赖
----

RPent、RLinf 和 RoboTwin 必须使用匹配的 ``adapt/robotwin-hybrid`` 分支。
RoboTwin 分支基于
``RLinf_support@0008ae6800df9f75fc8de7098bacb01735fd8fd2``，并包含
``compatibility/rpent_downloads_manifest.json`` 中记录的兼容补丁。
效果对齐运行不能替换成任意 RoboTwin checkout。

下载固定 revision 的 LingBot 权重：

.. code-block:: bash

   hf download RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500 \
      --revision c55199f25a10397e79dce177ee11c8774fb8edde \
      --local-dir /path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500

LingBot 源码目录必须提供 ``deploy.websocket_client_policy`` 和
``deploy.lingbot_vla_policy``。RPent 使用官方 websocket client/server，
不在 RPent 中重新实现 normalization。

运行
----

设置明确的源码和 Python 路径，然后启动单环境 hybrid episode：

.. code-block:: bash

   export ROBOTWIN_ROOT=/path/to/RoboTwin
   export ROBOTWIN_ASSETS_ROOT=/path/to/pinned/RoboTwin-assets
   export ROBOTWIN_CUROBO_ROOT=/path/to/curobo
   export RPENT_RLINF_ROOT=/path/to/RLinf
   export VLA_ROOT=/path/to/LingBot-VLA-source
   export LINGBOT_MODEL_PATH=/path/to/LingBot-VLA-RoboTwin-EEF-ckpt1500

   rpent --env robotwin \
      --task-name place_fan \
      --task-config demo_randomized \
      --seed 100002 \
      --seed-mode exact \
      --allow-infeasible \
      --instruction "$ROBOTWIN_INSTRUCTION" \
      --robotwin-hybrid-workspace "$ROBOTWIN_ROOT/hybrid_workspace" \
      --env-python /path/to/robotwin/python \
      --vla-python /path/to/lingbot/python \
      --env-cuda-device 0 \
      --vla-cuda-device 2 \
      --planner codex \
      --model gpt-5.5 \
      --reasoning-effort high

正式 Downloads parity 固定使用 ``demo_randomized``、exact native seed、
``--allow-infeasible`` 和冻结 seed 表中的 instruction。探索运行可以省略
``--instruction``，此时按选定 seed 确定性生成 native instruction。

``--allow-reset`` 才会启用同 seed 的受控 reset，默认关闭。
``--env-endpoint`` 和 ``--vla-endpoint`` 可以连接已有服务；效果 parity
应由 RPent 启动固定版本的本地服务，保证源码和模型身份可审计。
``--cuda-device`` 保留 Env/VLA 共用一张 GPU 的兼容行为，不能与
``--env-cuda-device`` 或 ``--vla-cuda-device`` 同时使用。正式 paired
运行会在 ``robotwin_resource_manifest.json`` 中记录两张物理 GPU 的
index 和 UUID。

``ROBOTWIN_ROOT`` 指向 compatibility 源码 checkout；
``ROBOTWIN_ASSETS_ROOT`` 指向单独固定、包含 ``assets/`` 的资产根目录。
完整打包的 checkout 可以让两者指向同一路径。
``ROBOTWIN_CUROBO_ROOT`` 必须是 clean 的
``NVlabs/curobo@2fbffc35225398cf9d5f382804faa9de2608753b`` checkout。
该固定 cuRobo revision 的 hybrid planner 还要求
``warp-lang==1.11.1``。
hybrid 启动会加载其 ``src`` 源码并拒绝其他 revision。
``--robotwin-hybrid-workspace`` 只由 RPent runtime 只读访问。runtime
仅在拒绝 legacy 协议标记、路径和 endpoint 后导入 allowlist 内的同 task
语义 recipe 字段；源 ``GUIDE.md``、memory 索引、保存过的坐标和
simulator oracle 状态都不会进入 Agent prompt。

运行契约
--------

启动握手固定为 ``robotwin-agent-v1``、``downloads_hybrid`` 和
``robotwin-rpent-downloads-2026-07-31-v1``。动作布局为 ``qpos14`` 和
世界坐标系的 ``eef16``，四元数顺序为 ``wxyz``。

每个修改环境状态的请求都有 server-scoped mutation ID。传输超时后，
RPent 查询原 ID，不会重放动作。native 状态不可信时会停止读取机械臂和
相机，只允许 status 和受控 reset。

任务成功只认 fresh ``TASK_ENV.eval_success``。VLA chunk 或 primitive
执行完成不代表任务成功。
