# RLHF Foundations

A small, readable codebase for learning RLHF on a Mac or Linux machine. The focus is on clear implementations of recent policy-optimization methods, a simple MoE-based transformer, and simple task environments—not on large-scale distributed training. Main purpose here is for people new to post-training and RLHF, to use this to learn basic concepts on their computer without needing GPUs. 

**Simulate Real-World Challenges:** Since the purpose of this repository is learning without access to a GPU cluster, one important feature here is to simulate certain bad behaviors that occur in practice. Currently, we simulate the issue that arise due to mismatch between log-probs in inference and training in LLMs which is specially bad for MoEs.

An accompanying PDF on RLHF foundations is planed for a later release this summer. If you have any specific request, then create an issue or email me.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_rl.py --config configs/dyck_grpo.yaml
```

The training script loads YAML config, applies optional CLI overrides (OmegaConf dot paths), runs supervised fine-tuning (SFT) when enabled, then RL. With visualization enabled, a live dashboard opens in your browser at `http://127.0.0.1:8050`.

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
  tasks/         # Task environments and tokenizers (Dyck language)
  utils/         # Pydantic configs and live Dash visualizer
configs/         # Experiment YAML files
scripts/         # Entry points (run_rl.py)
```

## Training pipeline

1. **Data** — Sample unique prompts from the task, split into train/val (`scripts/run_rl.py`).
2. **SFT** (optional) — Supervise on `target_completion` with an EOS token; reference model for KL is cloned *after* SFT.
3. **RL** — Batched rollout, reward, and policy update (GRPO-family objectives).
4. **Monitoring** — Losses, gradient norms, rewards, and sample generations in the Dash UI.

## RL algorithms

| Algorithm | Module | Reference |
|-----------|--------|-----------|
| GRPO | `rlhf/grpo.py` | [Paper](https://arxiv.org/pdf/2402.03300) |
| GSPO | `rlhf/gspo.py` | [Paper](https://arxiv.org/pdf/2507.18071) |
| CISPO | `rlhf/cispo.py` | [Paper](https://arxiv.org/pdf/2506.13585) |
| TIS | `rlhf/tis.py` | [Blog](https://fengyao.notion.site/off-policy-rl) |
| IcePop | `rlhf/icepop.py` | Importance-sampling variant |

Shared training logic lives in `rlhf/abstract_rl.py`. DPO and reward modeling are in `rlhf/dpo.py` and `rlhf/reward_modeling.py` for preference-style experiments.

Set the algorithm in config: `rl_config.algorithm: grpo` (also `gspo`, `cispo`, `tis`, `icepop`).

## Tasks

**Dyck language** (`tasks/dyck.py`) — Complete a partial bracket string `([{` with valid closings from `( )`, `[ ]`, `{ }`. Reward is 1 when the full string is balanced, 0 otherwise.

**Block arrangement** — Placeholder only (`tasks/block.py`).

Each task defines a tokenizer (with a required EOS token). Model `vocab_size` is taken from the task tokenizer, not from config.

## LLM

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

Pydantic validates config; `model_config` was renamed to `llm_config` because `model_config` is reserved by Pydantic.

## Visualization

`utils/visualize.py` runs a Plotly Dash app that tracks SFT/RL losses, GRPO/KL breakdown, gradient norms, train/eval rewards, and recent prompt/completion samples. Disable with `visualize.enabled: false` if you do not need it.

## Dependencies

`torch`, `pydantic`, `omegaconf`, `pyyaml`, `dash`, `plotly` — see `requirements.txt`.

## Contributing

Issues and pull requests are welcome.
