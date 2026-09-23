# MemBodied π₀ backend

This is the OpenPI-based π₀ implementation used for MemBodied. It contains the memory-token model, attention-steering and hierarchical variants, the initial-scene anchor, first-frame ablation, and the five-step LIBERO policy.

## Install

```bash
uv sync --frozen
```

## Commands

```bash
uv run scripts/compute_norm_stats.py --config-name membodied
uv run scripts/train.py membodied --exp-name EXPERIMENT
uv run scripts/serve_policy.py --config-name membodied --checkpoint-dir /path/to/checkpoint
```

Dataset IDs and ablations are CLI overrides. Available configs are `membodied`, `membodied_no_anchor`, `membodied_as`, `membodied_h`, `membodied_vision_only`, `membodied_action_only`, `membodied_first_frame`, and `membodied_libero`.

For RMBench, edit [deploy_policy.yml](deploy_policy.yml). The checkpoint can also be provided through `MEMBODIED_CHECKPOINT_DIR`.

