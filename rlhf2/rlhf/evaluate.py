import torch

from rlhf2.rlhf.inference import batch_generate_with_rewards


@torch.no_grad()
def evaluate_reward(
    model,
    eval_loader,
    tokenizer,
    reward_fn,
    *,
    temp: float,
    max_tokens: int,
    visualizer=None,
    log_step: int | None = None,
) -> float:
    """Mean task reward over the eval loader (one generation per prompt)."""
    model.eval()
    sum_rewards = 0.0
    num_samples = 0

    for data in eval_loader:
        _, mean_reward = batch_generate_with_rewards(
            data,
            model,
            tokenizer,
            reward_fn,
            temp=temp,
            max_tokens=max_tokens,
            K=1,
            visualizer=visualizer,
            log_step=log_step,
            log_phase="eval",
        )
        sum_rewards += mean_reward * len(data["prompt"])
        num_samples += len(data["prompt"])

    model.train()
    return sum_rewards / float(max(num_samples, 1))
