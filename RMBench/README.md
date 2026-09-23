# RMBench in MemBodied

This directory vendors the complete [RMBench](https://github.com/RoboTwin-Platform/RMBench) simulator and task set from commit `87e0498891073d483d330195c0f160709bd92ff5`. It is tracked as ordinary source, not as a Git submodule.

RMBench is built on RoboTwin 2.0. Its original MIT license is preserved in [LICENSE](LICENSE). Refer to the [RMBench project site](https://rmbench.github.io/) and upstream documentation for simulator assets, installation, data collection, and benchmark task details.

This distribution intentionally provides only the MemBodied policy integrations:

- [π₀ backend](policy/pi0/README.md)
- [π₀.₅ backend](policy/pi05/README.md)

Configure the matching `deploy_policy.yml`, activate that backend's locked environment, and invoke the normal RMBench evaluation command for the selected task. The common adapter fields are `backend`, `config_name`, `checkpoint_dir`, optional `asset_id`, and `action_chunk_size`.

## RMBench citation

```bibtex
@article{chen2026rmbench,
  title={RMBench: Memory-Dependent Robotic Manipulation Benchmark with Insights into Policy Design},
  year={2026},
  url={https://arxiv.org/abs/2603.01229}
}
```

