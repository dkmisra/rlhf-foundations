import torch


@torch.no_grad()
def evaluate(
    rollout_trainer,
    eval_loader,
    model,
    tokenizer,
    reward_fn,
    label: str = "Evaluation",
    visualizer=None,
    phase: str = "rl",
    step: int | None = None,
    sft_trainer=None,
):
    """Mean task reward over the eval loader (one generation per prompt).

    When ``phase == "sft"`` and ``sft_trainer`` is set, also logs mean teacher NLL on gold completions.
    """
    model.eval()
    device = next(model.parameters()).device
    sum_rewards = 0.0
    sum_teacher_nll = 0.0
    num_samples = 0
    log_phase = f"{phase}/eval" if step is not None else phase

    for data in eval_loader:
        _, mean_reward = rollout_trainer.collect_data(
            data,
            model,
            tokenizer,
            reward_fn,
            K=1,
            visualizer=visualizer,
            log_step=step,
            log_phase=log_phase,
        )
        sum_rewards += mean_reward * len(data["prompt"])
        num_samples += len(data["prompt"])

        if phase == "sft" and sft_trainer is not None:
            input_ids, attention_mask, response_mask = sft_trainer.prepare_data(data, tokenizer)
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            response_mask = response_mask.to(device)
            log_prob = sft_trainer.calc_log_prob(model, input_ids, attention_mask)
            mask = response_mask[:, 1:]
            teacher_nll = -(log_prob * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            sum_teacher_nll += teacher_nll.sum().item()

    mean_reward = sum_rewards / float(max(num_samples, 1))
    print(f"{label}: Mean reward is {mean_reward:.4f}")

    if visualizer is not None and step is not None:
        visualizer.log_eval_reward(mean_reward, phase, step)
        if phase == "sft" and sft_trainer is not None:
            mean_teacher_nll = sum_teacher_nll / float(max(num_samples, 1))
            visualizer.log_eval_teacher_nll(mean_teacher_nll, step)
            print(f"{label}: Mean teacher NLL is {mean_teacher_nll:.4f}")

    model.train()
    return mean_reward
