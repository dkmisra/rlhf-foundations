# RLHF Foundations

A small, readable codebase for learning RLHF on a Mac or Linux machine. The focus is on clear implementations of recent policy-optimization methods, a simple MoE-based transformer, and simple task environments—not on large-scale distributed training. Main purpose here is for people new to post-training and RLHF, to use this to learn basic concepts on their computer without needing GPUs. 

**Simulate Real-World Challenges:** Since the purpose of this repository is learning without access to a GPU cluster, one important feature here is to simulate certain bad behaviors that occur in practice. Currently, we simulate the issue that arise due to mismatch between log-probs in inference and training in LLMs which is specially bad for MoEs.

An accompanying PDF on RLHF foundations is planed for a later release this summer. If you have any specific request, then create an issue or email me.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_rl.py --config configs/dyck_grpo.yaml
```

The training script loads YAML config, applies optional CLI overrides (OmegaConf dot paths), runs supervised fine-tuning (SFT) when enabled, then RL. With visualization enabled, a live dashboard opens in your browser at `http://127.0.0.1:8050`. It will open a window and results will start filling in:

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
2. **SFT** (optional) — Supervise on `target_completion` with an EOS token; reference model for KL is cloned *after* SFT.
3. **RL** — Batched rollout, reward, and policy update (GRPO-family objectives).
4. **Monitoring** — Losses, gradient norms, rewards, and sample generations in the Dash UI.

Two sample tasks are provided. These tasks are chosen to be only slightly hard so they can be trained in 10-20min on a Macbook.

- **Dyck language** (`tasks/dyck.py`) — Complete a partial bracket string `([{` with valid closings from `( )`, `[ ]`, `{ }`. Reward is 1 when the full string is balanced, 0 otherwise.

- **Block arrangement** — Mirror image a sequence of blocks (e.g., red red blue green -> green blue red red) (`tasks/block.py`).



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

`llm/transformer.py` — Compact causal transformer with multi-head attention and RoPE (or absolute positions). `llm/moe.py` provides mixture-of-experts layers for studying train/inference mismatch (see `scripts/run_moe_mismatch.py`, stub).

## Configuration

Top-level config sections (see `utils/data_types.py`):

| Section | Purpose |
|---------|---------|
| `data_config` | Domain, dataset sizes, batching, Dyck sampling |
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

Issues and pull requests are welcome. Here are some future releases:

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
