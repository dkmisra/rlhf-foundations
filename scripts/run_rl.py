import argparse
import random
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rlhf2"))

from llm.transformer import Transformer
from rlhf.cispo import CISPO
from rlhf.grpo import GRPO
from rlhf.gspo import GSPO
from rlhf.icepop import IcePop
from rlhf.tis import TIS
from tasks.abstract import AbstractTask, AbstractTokenizer
from tasks.dyck import Dyck
from utils.data_types import Config, DataConfig
from utils.visualize import create_visualizer


ALGORITHMS = {
    "grpo": GRPO,
    "gspo": GSPO,
    "cispo": CISPO,
    "tis": TIS,
    "icepop": IcePop,
}


def get_task_tokenizer(task: AbstractTask) -> AbstractTokenizer:
    return type(task).get_tokenizer()


class BatchDict(dict):
    def __len__(self):
        return len(self["prompt"])


class TaskDataset(Dataset):
    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[str, dict]:
        item = self.items[idx]
        metadata = {k: v for k, v in item.items() if k != "prompt"}
        return item["prompt"], metadata


def collate_task_batch(batch: list[tuple[str, dict]]) -> BatchDict:
    prompts, metadata = zip(*batch)
    return BatchDict({"prompt": list(prompts), "metadata": list(metadata)})


def collect_unique_prompts(
    task: AbstractTask,
    num_unique: int,
    rng: random.Random,
    max_attempts: int,
    oversample: bool,
) -> list[dict]:
    """Sample unique prompts (by prompt string) from the task."""
    prompt_to_item: dict[str, dict] = {}
    attempts = 0

    while len(prompt_to_item) < num_unique and attempts < max_attempts:
        item = task.get_task()
        attempts += 1
        prompt = item["prompt"]
        if prompt not in prompt_to_item:
            prompt_to_item[prompt] = item

    unique_items = list(prompt_to_item.values())

    if len(unique_items) < num_unique:
        if not oversample:
            raise RuntimeError(
                f"Collected {len(unique_items)} unique prompts after {attempts} attempts, "
                f"but {num_unique} are required. Enable data_config.oversample to duplicate prompts."
            )
        unique_items = [rng.choice(unique_items) for _ in range(num_unique)]
    elif len(unique_items) > num_unique:
        unique_items = rng.sample(unique_items, num_unique)

    return unique_items


def build_train_val_splits(
    task: AbstractTask,
    data_config: DataConfig,
) -> tuple[list[dict], list[dict]]:
    """Sample unique prompts, then split into train and validation sets."""
    total = data_config.train_size + data_config.val_size

    rng = random.Random(data_config.seed)
    saved_rng = task.rng
    task.rng = random.Random(data_config.seed)

    unique_items = collect_unique_prompts(
        task,
        num_unique=total,
        rng=rng,
        max_attempts=data_config.max_sample_attempts,
        oversample=data_config.oversample,
    )
    task.rng = saved_rng

    rng.shuffle(unique_items)
    train_items = unique_items[: data_config.train_size]
    val_items = unique_items[data_config.train_size : total]
    return train_items, val_items


def build_dataloader(
    items: list[dict],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TaskDataset(items),
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_task_batch,
        generator=generator if shuffle else None,
    )


def load_config(config_path: Path, overrides: list[str]) -> Config:
    yaml_cfg = OmegaConf.load(config_path)
    if overrides:
        yaml_cfg = OmegaConf.merge(yaml_cfg, OmegaConf.from_dotlist(overrides))
    return Config.model_validate(OmegaConf.to_container(yaml_cfg, resolve=True))


def build_task(data_config):
    if data_config.domain == "dyck":
        return Dyck(
            mean_prompt_length=data_config.mean_prompt_length,
            prompt_length_min=data_config.prompt_length_min,
            prompt_length_max=data_config.prompt_length_max,
            max_completion_length=data_config.max_completion_length,
            rng=random.Random(data_config.seed),
        )
    raise ValueError(f"Unknown domain: {data_config.domain!r}")


def build_model(llm_config, tokenizer: AbstractTokenizer, device: str) -> Transformer:
    model = Transformer(llm_config, tokenizer.vocab_size)
    return model.to(device)


def build_trainer(rl_config):
    algorithm_cls = ALGORITHMS[rl_config.algorithm]
    return algorithm_cls(rl_config)


def make_reward_fn(task):
    def reward_fn(prompt: str, metadata: dict, completion: str) -> float:
        del metadata
        return task.compute_reward(prompt, completion)

    return reward_fn


def run_experiment(config: Config) -> None:
    device = config.device
    task = build_task(config.data_config)
    reward_fn = make_reward_fn(task)

    train_items, val_items = build_train_val_splits(task, config.data_config)
    train_loader = build_dataloader(
        train_items,
        config.data_config.batch_size,
        shuffle=True,
        seed=config.data_config.seed,
    )
    eval_loader = build_dataloader(
        val_items,
        config.data_config.batch_size,
        shuffle=False,
        seed=config.data_config.seed + 1,
    )

    tokenizer = get_task_tokenizer(task)
    model = build_model(config.llm_config, tokenizer, device)
    trainer = build_trainer(config.rl_config)
    visualizer = create_visualizer(config.visualize)

    print(f"Launching {config.rl_config.algorithm} on {config.data_config.domain} (vocab_size={tokenizer.vocab_size})")
    if visualizer is not None:
        print(f"Dashboard: http://{config.visualize.host}:{config.visualize.port}")
    print(
        f"Unique prompts: train={len(train_items)}, val={len(val_items)} "
        f"(target {config.data_config.train_size}/{config.data_config.val_size})"
    )
    trainer.train(model, train_loader, eval_loader, tokenizer, reward_fn, visualizer)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RLHF training from YAML config")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "dyck_grpo.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Dot-path overrides merged into config, e.g. rl_config.lr=0.001 data_config.seed=1",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config, args.overrides)
    run_experiment(config)


if __name__ == "__main__":
    main()
