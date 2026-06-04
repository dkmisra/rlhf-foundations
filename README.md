# RLHF Foundations: Try RLHF on your Local Macbook

This is a small compact codebase for learning RLHF concepts that you can run on your Mac or Linux machine. Main purpose here is for people new to post-training and RLHF, to use this to learn basic concepts on their computer without needing GPUs. The code contains recent policy-optimization methods, a simple MoE-based transformer, and simple tasks environments that you can train on in 10-20min. 

**Simulate Real-World Challenges:** Since the purpose of this repository is learning without access to a GPU cluster, one important feature this codease provides is to simulate certain bad behaviors that occur in practice. Presently, we simulate the issue that arise due to mismatch between log-probs in inference and training in LLMs which is specially bad for MoEs.

**Accompanying Book:** An accompanying PDF on RLHF foundations is planned for a later release this summer. If you have any specific request, then create an issue or email me.

**This is a beta release (June-4-2026)**: Some features maybe broken. A stable release will come in under a week.

## Quick start

Install the requirements, and run the code on a sample YAML file as shown:

```bash
pip install -r requirements.txt
python scripts/run_rl.py --config configs/dyck_grpo.yaml
```

The training script loads YAML config, applies optional CLI overrides (OmegaConf dot paths), runs supervised fine-tuning (SFT) when enabled, and then RL. With visualization enabled, a live dashboard opens in your browser at `http://127.0.0.1:8050`. It will open a window and results will start filling in as shown:

![visualization of results](img/visual.png)

Override any field from the command line:

```bash
python scripts/run_rl.py --config configs/dyck_grpo.yaml \
  rl_config.algorithm=gspo \
  rl_config.max_epochs=5 \
  data_config.train_size=32 \
  visualize.enabled=false
```

## Repository layout

```
rlhf2/
  rlhf/          # RL trainers (GRPO, GSPO, CISPO, TIS, IcePop), SFT, evaluation
  llm/           # Transformer, RoPE, MoE building blocks
  tasks/         # Task environments and tokenizers (Dyck language, Block mirroring)
  utils/         # Pydantic configs and live Dash visualizer
configs/         # Experiment YAML files
scripts/         # Entry points (run_rl.py)
```

1. **Data** — Sample unique prompts from the task, split into train/val (`scripts/run_rl.py`).
2. **SFT** (optional) — SFT is performed prior to RL by default.
3. **RL** — Batched rollout, reward, and policy update (GRPO-family objectives).
4. **Monitoring** — Losses, gradient norms, rewards, and sample generations in the Dash UI.

Two sample tasks are provided. These tasks are chosen to be only slightly hard so they can be trained in 10-20min on a Macbook.

- **Dyck language** (`tasks/dyck.py`) — Complete a partial bracket string `([{` with valid closings from `( )`, `[ ]`, `{ }`. Reward is 1 when the full string is balanced, 0 otherwise.

- **Block arrangement** — Mirror image a sequence of blocks (e.g., red red blue green -> green blue red red) (`tasks/block.py`).

More tasks maybe added in the future.

## RL algorithms

| Algorithm | Module | Reference |
|-----------|--------|-----------|
| GRPO | `rlhf/grpo.py` | [Paper](https://arxiv.org/pdf/2402.03300) |
| GSPO | `rlhf/gspo.py` | [Paper](https://arxiv.org/pdf/2507.18071) |
| CISPO | `rlhf/cispo.py` | [Paper](https://arxiv.org/pdf/2506.13585) |
| TIS | `rlhf/tis.py` | [Blog](https://fengyao.notion.site/off-policy-rl) |
| IcePop | `rlhf/icepop.py` | [Blog](https://ringtech.notion.site/icepop) |

Shared training logic lives in `rlhf/abstract_rl.py`. DPO and reward modeling are in `rlhf/dpo.py` and `rlhf/reward_modeling.py` for preference-style experiments.

Set the algorithm in config: `rl_config.algorithm: grpo` (also `gspo`, `cispo`, `tis`, `icepop`).

## LLM Implementation

`llm/transformer.py` contains a compact causal transformer with multi-head attention and RoPE (or absolute positions). The file `llm/moe.py` provides mixture-of-experts layers and `llm/experts.py` provides list of expert models -- currently, MLP and SwishGLU.

## Configuration

Top-level config sections (see `utils/data_types.py`):

| Section | Purpose |
|---------|---------|
| `data_config` | Dataset sizes, batching, and domain specific hyperparameters |
| `llm_config` | Transformer shape (layers, dim, heads, `max_seq`) |
| `rl_config` | Algorithm, optimization, `K`, clipping, KL, `inference`, `sft` |
| `visualize` | Live dashboard (port, logging frequency, generations table) |
| `device` | e.g. `cpu` or `cuda` |

OmegaConf is used to load configurations from yaml and command line, and pydantic is used for validation.

## Visualization

`utils/visualize.py` runs a Plotly Dash app that tracks SFT/RL losses, GRPO/KL breakdown, gradient norms, train/eval rewards, and recent prompt/completion samples. Disable with `visualize.enabled: false` if you do not need it.

## Dependencies

The repository relies on basic packages such as `torch`, `pydantic`, `omegaconf`, `pyyaml`, `dash`, and `plotly`. See `requirements.txt` for up to date list.

## Contributing

Issues and pull requests are welcome. Here are features I'd like to add in the future:

1. More tasks that are simple enough to be trained in under 20min on a Macbook Pro
2. Tasks that require some simplistic version of reasoning and using process reward models for solving these.
3. Saving and resuming sessions.

If you find this work useful, you can cite the following:

```bibtex
@misc{misra2026rlhffoundations,
  author       = {Misra, Dipendra},
  title        = {RLHF2: Run and Understand RLHF Concepts on Your Macbook},
  year         = {2026},
  howpublished = {\url{https://github.com/dkmisra/rlhf-foundations}},
  note         = {Educational codebase for learning RLHF on a single machine}
}
```
