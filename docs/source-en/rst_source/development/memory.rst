Memory Management
=================

This memory lets the agent accumulate what works and what doesn't across runs
and reuse it across tasks — in particular, which prompts and conditions make
the VLA reliable, so it need not rediscover them from scratch each run.

RPent's global memory on LIBERO lives under ``resources/libero/memory/``: a
``MEMORY.md`` serves as the index, with a handful of standalone notes beneath
it. It is a reviewed, read-only knowledge base, read at the start of each run.

Hosting
-------

``resources/`` is not vendored in git. It is hosted on the HuggingFace dataset
``RLinf/RPent-memory`` (laid out per environment, e.g. ``libero/memory/`` and
``libero/results_*_pert/``). ``rpent.utils.resources.ensure_resources`` syncs the
env's subtree from the dataset on each run (incremental: only changed files are
downloaded), so the local copy stays up to date. The dataset is public, so a
fresh clone downloads it without a token. Set ``HF_HUB_OFFLINE=1`` to skip the
sync and use the local copy only. Memory is optional: if an env has none on the
dataset, or the sync fails, the run continues with whatever is on disk.

Updating the memory
-------------------

Publishing memory is gated to maintainers with write access to the ``RLinf``
organisation; the repository ships no self-serve upload path. To contribute a
better memory entry, open an issue with the proposed content and a maintainer
will review and publish it.
