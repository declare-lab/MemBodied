<p align="center">
  <img src="assets/membodied-wordmark.svg" alt="MemBodied" width="460">
</p>

<h3 align="center">Recurrent Associative Memory for Vision-Language-Action Models</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2609.28256"><strong>Paper</strong></a> &nbsp;·&nbsp;
  <a href="https://declare-lab.github.io/MemBodied/"><strong>Project & demonstrations</strong></a> &nbsp;·&nbsp;
  <a href="#getting-started"><strong>Getting started</strong></a> &nbsp;·&nbsp;
  <a href="#citation"><strong>Citation</strong></a>
</p>

<p align="center">
  Tej Deep Pala · Navonil Majumder · Bryce Goh · Raphael Yee<br>
  Jianfei Yang · Liming Chen · Soujanya Poria
</p>

---

**Read from memory. Write from experience.** MemBodied gives a vision-language-action policy a persistent episodic state. Gated associative reads bring past interactions into the action expert; delayed writes associate each action with its observed outcome. A separate initial-scene anchor preserves a reference to where the episode began.

This is the official implementation for [the paper](https://arxiv.org/abs/2609.28256), including the π₀ and π₀.₅ RMBench backends, memory ablations, and the π₀ LIBERO configuration.

## Results at a glance

| Evaluation | Comparison | Relative success | Mean success: baseline → MemBodied |
|:--|:--|--:|--:|
| RMBench · five tasks | Stateless π₀ | **7.81×** | 6.4% → 50.0% |
| RMBench · five tasks | Vanilla recurrent memory | **2.98×** | 16.8% → 50.0% |
| RMBench · π₀.₅ backbone | π₀.₅ baseline | **3.87×** | 12.4% → 48.0% |
| Physical robots · three tasks | π₀ baseline | **8×** | 3.33% → 26.67% |
| LIBERO-Long | π₀ | **1.06×** | 85.2% → 90.6% |

Multipliers compare mean task-success rates. Vanilla recurrent memory is the strongest **non-video stateful baseline** in the paper's π₀ RMBench evaluation. Physical-robot results are 16 successful trials versus 2, out of 60 per policy. Task-level results and evaluation protocols are available on the [project page](https://declare-lab.github.io/MemBodied/#results).

**Alongside video-history memory.** NativeMEM is reported separately: it achieves 38.4% mean success on RMBench, rising to 45.2% when combined with MemBodied. Standalone MemBodied achieves 50.0%. No method leads on every task.

## Method

<p align="center">
  <img src="assets/architecture.png" alt="MemBodied architecture: associative read, action generation, initial-scene anchor, and transition-conditioned memory write" width="100%">
</p>

<p align="center"><em>Read before acting. Update memory after observing the outcome.</em></p>

- **Associative memory.** The current policy state queries layer-wise matrices. A gated readout enters a dedicated memory token and informs action generation through self-attention.
- **Episode anchor.** A compact representation of the first observation remains fixed, providing a reference for details such as an object's original location.
- **Transition-conditioned writes.** After execution, the action chunk and its observed consequence update the matrices through a learned gated delta rule.

The memory pathways learn through the policy's native action objective. Episodic storage is **O(1) in episode length**, at a fixed architecture, rank, and camera configuration. Its contents change as the robot interacts; its capacity stays fixed. Bounded frame stacking can also use constant storage—the contribution is how memory is read, written, and supplied to the policy.

## Getting started

### 1. Clone and install

```bash
git clone https://github.com/declare-lab/MemBodied.git
cd MemBodied
```

Follow the [RMBench setup guide](RMBench/README.md) for simulator installation and assets. The two policy backends are separate Python projects, each with a lockfile. Select one environment:

```bash
# π₀ backend
cd RMBench/policy/pi0
uv sync --frozen
```

For π₀.₅, use `RMBench/policy/pi05` instead. The backends require Python 3.11 or later; their project files specify the platform-dependent JAX dependencies. Benchmark assets, datasets, and trained checkpoints are not bundled in this repository.

### 2. Configure data and train

Set the dataset repository ID in your selected backend's `src/openpi/training/config.py` before computing normalization statistics. The RMBench configurations initially use `YOUR_RMBENCH_DATASET` as a placeholder. Keep the dataset selection consistent between normalization and training.

From `RMBench/policy/pi0`:

```bash
uv run scripts/process_data.py --help
uv run scripts/compute_norm_stats.py --config-name membodied
uv run scripts/train.py membodied --exp-name EXPERIMENT
```

Training accepts overrides such as `--data.repo-id ORG/DATASET`, `--model.sequence-len 14`, and `--model.memory-rank 64`. The normalization script reads the registered configuration directly. For π₀.₅, run the corresponding commands in its backend with `membodied_pi05`.

RMBench sequence lengths are 8 for Put Back Block and Rearrange Blocks, 14 for Battery Try and Swap Blocks, and 18 for Block Ranking. The default associative rank is 128 and the memory scale is 256.

### 3. Evaluate

Configure the selected backend's `deploy_policy.yml`:

```yaml
backend: pi0             # pi0 or pi05
config_name: membodied   # membodied_pi05 for the π₀.₅ backend
checkpoint_dir: /path/to/checkpoint
asset_id: null
action_chunk_size: 50
```

Activate the matching backend environment and follow the [RMBench evaluation instructions](RMBench/README.md). You can also provide the checkpoint through `MEMBODIED_CHECKPOINT_DIR`.

For LIBERO, use the [dedicated example](RMBench/policy/pi0/examples/libero/README.md). The `membodied_libero` configuration uses sequence length 6 during training and ten interleaved recurrent slots for five-step replanning.

## Configurations and code

| Path | Contents |
|:--|:--|
| [`RMBench/`](RMBench/) | Simulator, tasks, and evaluation entry points |
| [`RMBench/policy/pi0/`](RMBench/policy/pi0/) | π₀ implementation, ablations, and LIBERO integration |
| [`RMBench/policy/pi05/`](RMBench/policy/pi05/) | π₀.₅ implementation |

<details>
<summary><strong>Available model configurations</strong></summary>

| Configuration | Setting |
|:--|:--|
| `membodied` | Memory-token model with associative memory and episode anchor |
| `membodied_no_anchor` | Memory-token model without the anchor |
| `membodied_as` | Attention-steering variant |
| `membodied_h` | Hierarchical variant with an LSTM cell |
| `membodied_vision_only` | Vision-only memory values, without the anchor |
| `membodied_action_only` | Action-only memory values, without the anchor |
| `membodied_first_frame` | Full-resolution first-frame ablation |
| `membodied_pi05` | π₀.₅ MemBodied |
| `membodied_libero` | π₀ LIBERO configuration |

</details>

## Citation

If you use MemBodied in your research, please cite:

```bibtex
@misc{pala2026membodied,
  title         = {{MemBodied}: Recurrent Associative Memory for Vision-Language-Action Models},
  author        = {Pala, Tej Deep and Majumder, Navonil and Goh, Bryce and Yee, Raphael and
                   Yang, Jianfei and Chen, Liming and Poria, Soujanya},
  year          = {2026},
  eprint        = {2609.28256},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  doi           = {10.48550/arXiv.2609.28256},
  url           = {https://arxiv.org/abs/2609.28256}
}
```

## License and acknowledgments

MemBodied is released under the [Apache 2.0 license](LICENSE). This implementation builds on [OpenPI](https://github.com/Physical-Intelligence/openpi), [RMBench](https://github.com/RoboTwin-Platform/RMBench), and [LeRobot](https://github.com/huggingface/lerobot). Vendored components retain their original licenses; see [third-party notices](THIRD_PARTY_NOTICES.md).
