import torch


@torch.no_grad()
def evaluate_reward(
    rollout_trainer,
    eval_loader,
    model,
    tokenizer,
    reward_fn,
    label: str = "Evaluation",
    visualizer=None,
    phase: str = "rl",
    step: int | None = None,
):
    """Mean task reward over the eval loader (one generation per prompt)."""
    model.eval()
    sum_rewards = 0.0
    num_samples = 0

    for data in eval_loader:
        _, mean_reward = rollout_trainer.collect_data(
            data,
            model,
            tokenizer,
            reward_fn,
            K=1,
            visualizer=visualizer,
            log_step=step,
            log_phase=phase,
        )
        sum_rewards += mean_reward * len(data["prompt"])
        num_samples += len(data["prompt"])

    mean_reward = sum_rewards / float(max(num_samples, 1))
    print(f"{label}: Mean reward is {mean_reward:.4f}")

    if visualizer is not None and step is not None:
        visualizer.log_eval_reward(mean_reward, phase, step)

    model.train()
    return mean_reward
