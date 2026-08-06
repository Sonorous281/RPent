# Copyright 2026 The RPent Authors.

"""User prompt for one RoboTwin run."""

CELL = """- task: {{task_name}}
- requested_seed: {{seed}}
- initial_native_seed: {{initial_seed}}
- seed_mode: {{seed_mode}}
- task_config: {{task_config}}
- allow_infeasible: {{allow_infeasible}}
- checkpoint: RLinf/LingBot-VLA-RoboTwin-EEF-ckpt1500
- checkpoint_revision: c55199f25a10397e79dce177ee11c8774fb8edde"""

SEMANTIC_RECIPE = """{{hybrid_context}}"""

BEGIN = """Inspect view_driver_state(step=0) and the returned images. Use only
registered RoboTwin tools to act. Copy the complete current task_language for
every lingbot_act. The native success predicate and action budget are
authoritative."""
