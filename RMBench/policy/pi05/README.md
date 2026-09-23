# MemBodied π₀.₅ backend

This is the independent OpenPI-based π₀.₅ implementation used for the RMBench MemBodied experiment.

## Install and run

```bash
uv sync --frozen
uv run scripts/compute_norm_stats.py --config-name membodied_pi05
uv run scripts/train.py membodied_pi05 --exp-name EXPERIMENT
uv run scripts/serve_policy.py --config-name membodied_pi05 --checkpoint-dir /path/to/checkpoint
```

Set the dataset repository with a CLI override. For RMBench evaluation, edit [deploy_policy.yml](deploy_policy.yml); `MEMBODIED_CHECKPOINT_DIR` is also accepted.

