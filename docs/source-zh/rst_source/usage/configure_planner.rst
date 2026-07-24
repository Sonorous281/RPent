Agentic Planner
===============

RPent 的推理大脑，也就是 planner，用一个命令行参数来选：

.. code-block:: bash

   --planner {api, claude_code, codex}

三种 planner 看到的是同一份 tool schema 和同一份 prompt bundle，区别只在于工具
调用循环 *怎么* 编排，以及各自能接哪些 LLM 或 SDK。

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - ``--planner``
     - 它是什么
     - 什么时候选它
   * - ``api``
     - 基于 `pydantic-ai <https://ai.pydantic.dev/>`_ 的与 provider
       无关的 tool-calling 循环。支持 Anthropic、OpenAI Responses、
       OpenAI 兼容 chat 接口，内置 prompt 缓存和历史图片剪枝。
     - 需要最细的调用控制、最广的 provider 覆盖，或者每轮开销最省的场景。
   * - ``claude_code``
     - `Claude Agent SDK
       <https://docs.claude.com/en/api/agent-sdk/overview>`_。
       把 RPent 的 toolkit 暴露成一个进程内的 MCP server，由 Claude
       驱动循环。
     - 想用 Claude 原生的 agent 运行时，比如记忆、thinking-mode 预算、
       健壮的工具重试。
   * - ``codex``
     - OpenAI **Codex SDK**，通过 HTTP MCP server 桥接到 toolkit。
     - 想用 Codex 的 agent 运行时，或者手头已经有 OpenAI、Codex 的
       配额可用。

``api`` planner（自定义、轻量）
--------------------------------

``--planner api`` 跑的是一段手写的 pydantic-ai 循环。它是默认值，可移植性也最好，
凡是讲 Anthropic Messages API、OpenAI Responses API，或 OpenAI 兼容 chat API 的
提供方都能接。

用 ``--model`` 的前缀来选提供方:

.. code-block:: bash

   # Anthropic Claude
   rpent --planner api --model anthropic:claude-opus-4-8 ...

   # OpenAI Responses (例如 GPT-5.5)
   rpent --planner api --model openai:gpt-5.5 ...

   # OpenAI 兼容 chat (例如 GLM 5.2, 纯文本)
   rpent --planner api --model openai-chat:glm-5.2 --no-images ...

它读取的环境变量 (需要覆盖时用 ``--base-url``):

- ``anthropic:*`` → ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_API_KEY``
- ``openai:*`` / ``openai-chat:*`` → ``OPENAI_BASE_URL`` /
  ``OPENAI_API_KEY``

``api`` 专属的调节参数：

- ``--max-tokens`` 是单次 LLM 回复的 token 上限，默认 ``8192``。
- ``--max-turns`` 是工具调用的轮数上限，默认 ``100``。
- ``--no-images`` 让 RPent 不向模型发送图片字节。纯文本模型必须加这个参数，
  否则会报 ``400 "message type 'image_url' is not supported"``。加了之后，
  智能体只能靠文本状态推理，任务表现可能不够理想。

``claude_code`` planner
------------------------

``--planner claude_code`` 把循环交给 Claude Agent SDK 来跑。RPent 的工具会变成
一个进程内的 MCP server 供 Claude Code 直接调用，在它眼里工具名都带着
``mcp__rpent__<name>`` 命名空间前缀。

.. code-block:: bash

   rpent --planner claude_code \
     --model claude-opus-4-8 \
     --suite libero_object_swap --task 2 --seed 0

注意事项：

- ``--model`` **不要** 加 provider 前缀，直接写 ``claude-opus-4-8`` 即可。
- 子进程有 wall-clock 上限 (``--planner-timeout-s``，默认取
  ``CELL_TIMEOUT_S`` / ``1200``)。
- 通过 ``--claude-code-max-budget-usd`` 设置美元预算 (默认取
  ``MAX_BUDGET_USD`` 环境变量或 ``10``)。
- Claude Code 需要单独安装和登录，见
  `Claude Agent SDK 文档
  <https://docs.claude.com/en/api/agent-sdk/overview>`_。

``codex`` planner
------------------

``--planner codex`` 通过 ``scripts/codex_proxy/`` 起的 HTTP MCP server
把同一个 toolkit 桥接到 OpenAI Codex SDK。

.. code-block:: bash

   rpent --planner codex \
     --model gpt-5.5 \
     --suite libero_goal_task --task 1 --seed 0

注意事项：

- ``--planner-timeout-s`` 的语义与 ``claude_code`` 相同。
- Codex 用标准的 OpenAI 环境变量做认证。

自带 agent
----------

如果这三种 planner 都不合适，比如你想接入内部的 planner、一个实验性的研究原型，
或者另一套 agent SDK，那就继承 ``rpent.planner.base.Planner``，并在
``rpent.planner.base.build_planner`` 里加一个分支来构造它：

.. code-block:: python

   # rpent/planner/mybrain.py
   from rpent.planner.base import Planner, PlannerResult

   class MyPlanner(Planner):
       def solve(self, *, system_prompt: str, user_message: str,
                 toolkit, max_turns: int, input_queue=None) -> PlannerResult:
           # 自己驱动 tool-calling 循环。
           # tools_spec = toolkit.get_tools_spec()   # LLM 看到的 tool schema
           # result = toolkit.execute_tool(name, input_dict)  # 返回一个 ToolResult
           ...

任何 planner 必须：

1. 接收已经渲染好的 ``system_prompt`` 和 ``user_message`` 字符串。CLI 会在调用
   ``solve`` 之前，从 ``robots/<env>/prompt_bundle.py`` 渲染出这两段文本。
2. 循环处理 LLM 的回复，抽出其中的工具调用，通过
   ``toolkit.execute_tool(name, input_dict)`` 转发给 toolkit，工具的 schema 则
   来自 ``toolkit.get_tools_spec()``。
3. 把每个工具的返回值以多模态上下文 (文本加图像) 的形式喂回 LLM。
4. 遇到 ``finish`` 或达到上限时终止。

因为所有 planner 看到的是同一份 schema 和 prompt，新增一个大脑不用改动工具或
env server。接口参见 :doc:`../development/architecture`；想给自定义大脑暴露新
工具，见 :doc:`../development/add_primitive`。

选择 max-tokens 与 max-turns
----------------------------

有两个参数圈定每次 planner 运行的规模。

- ``--max-tokens`` 限制 *每次回复* 的 token 数。LIBERO 这类任务通常 ``8192``
  就够，更长时序的 RoboCasa episode 如果模型支持可以调大。
- ``--max-turns`` 限制 *工具调用的总轮数*。单个 LIBERO 任务通常不超过 30 轮，
  RoboCasa 的长时序任务可能接近默认的 ``100``。

两个上限触达时都会以 ``finish(stuck)`` 优雅收尾，不会硬崩，所以可以放心调参，
transcript 不会丢。
