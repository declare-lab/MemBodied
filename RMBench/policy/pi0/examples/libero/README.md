# MemBodied on LIBERO

This directory contains the LIBERO dataset converter and evaluation client for `membodied_libero`.

Install the π₀ backend with `uv sync --frozen`. Install LIBERO in a separate compatible evaluation environment from its upstream repository, then install `packages/openpi-client` into that environment.

Convert demonstrations using:

```bash
uv run examples/libero/convert_libero_data_to_lerobot.py --help
```

Compute statistics, train, and start the policy server:

```bash
uv run scripts/compute_norm_stats.py --config-name membodied_libero
uv run scripts/train.py membodied_libero --exp-name EXPERIMENT
uv run scripts/serve_policy.py \
  --config-name membodied_libero \
  --checkpoint-dir /path/to/checkpoint \
  --asset-id libero
```

From the LIBERO environment, run `python examples/libero/main.py --replan-steps 5`. The policy uses ten recurrent slots in round-robin order, so each slot is revisited after 50 environment steps while actions are replanned every five steps. Reset the policy between episodes.

