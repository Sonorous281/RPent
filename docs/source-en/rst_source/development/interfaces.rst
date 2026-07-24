Core interfaces
===============

When you extend RPent — adding an environment, a planner, or a
transport — you implement one of a few small, stable interfaces. This
page collects them in one place. For the step-by-step how-tos, see
:doc:`add_robot` and :doc:`add_primitive`; for the big-picture design,
see :doc:`architecture`.

Environment entry point
-----------------------

Passing ``--env myenv`` makes ``rpent/envs/base.py`` import the
``robots.myenv`` package on demand and call the two factories it
exposes:

.. code-block:: python

   # robots/myenv/__init__.py
   def get_env_spec() -> EnvSpec: ...
   def get_toolkit(*, primitives_kwargs, video_path=None, dashboard=None): ...

There is **no central list** of envs — dropping a package under
``robots/`` is enough. ``get_env_spec`` returns the env identity and its
prompt bundle; ``get_toolkit`` builds the toolkit the planner drives.

Planner interface
-----------------

Every planner implements the same tiny interface (see
``rpent.planner.base``):

- Accept already-rendered ``system_prompt`` and ``user_message``
  strings (the CLI renders them from the env's prompt bundle before
  calling ``solve``).
- Accept a ``toolkit`` (which exposes tool schemas via
  ``get_tools_spec()`` and dispatches calls via
  ``execute_tool(name, input_dict)``).
- Drive the tool-calling loop.
- Feed each tool result back as multimodal context.
- Terminate on ``finish`` or when caps are hit.

That is the entire abstraction. The three built-in planners differ
only in *how* they meet the contract — see
:doc:`../usage/configure_planner` for the user-facing view and
``rpent/planner/api_loop.py`` / ``claude_code.py`` / ``codex.py``
for the code.

Toolkit interface
-----------------

A toolkit (``rpent.tools.toolkit.Toolkit``) owns:

- A **primitive driver** — a plain Python object that holds the env
  client, the VLA client, and any per-run state. Each tool the LLM
  can call corresponds to a method on this object.
- A set of **tool schemas** in Anthropic shape (``name``,
  ``description``, ``input_schema``), registered via
  ``self.add_tool(name, spec, handler)``.
- A per-step **state dump** — every primitive tool re-renders the
  world after it runs, so the next ``view_driver_state`` call sees
  the post-action state.

The base class also handles video capture (``episode.mp4``) and the
dashboard event stream. Any new env's ``toolkit.py`` subclasses this
class and registers whatever tools that env exposes.

Transport substrate
-------------------

Two codecs are supported natively, selected via the server's
``--transport {http,socket}`` flag (default ``http``) and mirrored on
the client side by ``--env-endpoint`` / ``--vla-endpoint`` protocol
prefix:

- **HTTP** (``rpent.utils.http_rpc``) — JSON body over ``POST /call``.
  Convenient for standard load balancing and cross-language clients.
  Numpy arrays cross the wire tagged as
  ``{"__ndarray__": <base64>, "dtype": ..., "shape": [...]}``.
- **Pickle-framed socket RPC** (``rpent.utils.socket_rpc``) — for
  history-stacked nested numpy dicts and other wide, variable-shape
  payloads where JSON re-encoding is wasteful.

Server-side, subclass :class:`rpent.utils.rpc.RpcFacade` and implement
``_dispatch(method, args, kwargs)``; the base provides shutdown, healthz,
transport binding, parent-death watch, and clean teardown. Adding a new
transport is a matter of implementing the two-method ``RpcClient``
interface (``call(method, args, kwargs, timeout_s)``); the toolkit and
planner stay unchanged.
