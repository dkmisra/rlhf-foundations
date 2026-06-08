"""Shared rollout utilities: sampling completions and scoring them with rewards.

These are plain functions (not trainer methods) so any stage -- SFT, RL, or
evaluation -- can generate rollouts without needing an RL-trainer instance.
"""

import torch


@torch.no_grad()
def batch_generate(model, prompts: list[str], tokenizer, *, temp: float, max_tokens: int, K: int = 1) -> list[list[str]]:
    """Batched autoregressive sampling. Returns K completions per prompt."""

    if not prompts:
        return []

    device = next(model.parameters()).device
    pad_id = tokenizer.PAD_ID
    eos_id = tokenizer.eos
    model.eval()

    expanded_prompts = [prompt for prompt in prompts for _ in range(K)]
    inputs = tokenizer(expanded_prompts, padding=True, return_tensor="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    batch_size = input_ids.size(0)

    kv_cache = None
    out_tokens = []

    for i in range(max_tokens):

        logits, kv_cache = model(input_ids, attention_mask, kv_cache)

        # We assume right-padding here, so need to do some work to extract last_logits
        if i == 0:
            prompt_len = attention_mask.sum(1) - 1                     # batch
            prompt_len = prompt_len.long()
            prompt_len = prompt_len.view(-1, 1, 1)                     # batch x 1 x 1
            prompt_len = prompt_len.expand(-1, -1, logits.shape[2])                 # batch x 1 x vocab
            last_logits = torch.gather(logits, index=prompt_len, dim=1).squeeze(1)  # batch x dim
        else:
            last_logits = logits[:, -1, :]                                          # batch x dim

        probs = torch.softmax(last_logits / max(temp, 1e-6), dim=-1)
        next_tokens = torch.multinomial(probs, num_samples=1)

        input_ids = next_tokens                                                          # batch x 1
        attention_mask = torch.cat([attention_mask, torch.ones_like(input_ids)], dim=1)  # batch x (kv_len + 1)
        out_tokens.append(next_tokens)

    out_tokens = torch.cat(out_tokens, dim=1)           # batch x gen_len

    completions: list[str] = []
    for i in range(batch_size):
        gen_ids: list[int] = []
        for token_id in out_tokens[i].tolist():
            if token_id == pad_id:
                break
            if token_id == eos_id:
                break
            gen_ids.append(token_id)
        completions.append(tokenizer.decode(gen_ids))

    return [completions[i * K : (i + 1) * K] for i in range(len(prompts))]


def batch_generate_with_rewards(
    data,
    model,
    tokenizer,
    reward_fn,
    *,
    temp: float,
    max_tokens: int,
    K: int,
    visualizer=None,
    log_step: int | None = None,
    log_phase: str = "train",
):
    """Generate K completions per prompt and score them with ``reward_fn``.

    Returns ``(batch_with_gens_and_rewards, mean_reward)`` where each entry of
    the first item is ``({"prompt", "metadata"}, generations, rewards)``.
    """

    prompts = data["prompt"]
    metadata_list = data["metadata"]

    all_generations = batch_generate(model, prompts, tokenizer, temp=temp, max_tokens=max_tokens, K=K)

    batch_with_gens_and_rewards = []
    sum_reward = 0.0
    for prompt, metadata, generations in zip(prompts, metadata_list, all_generations):
        rewards = [reward_fn(prompt, metadata, generation) for generation in generations]
        sum_reward += sum(rewards)
        batch_with_gens_and_rewards.append(
            ({"prompt": prompt, "metadata": metadata}, generations, rewards)
        )

    if visualizer is not None and log_step is not None:
        visualizer.log_rollout_batch(
            step=log_step,
            phase=log_phase,
            prompts=prompts,
            metadata_list=metadata_list,
            generations_per_prompt=all_generations,
            rewards_per_prompt=[item[2] for item in batch_with_gens_and_rewards],
        )

    mean_reward = sum_reward / float(len(prompts) * K)
    return batch_with_gens_and_rewards, mean_reward
