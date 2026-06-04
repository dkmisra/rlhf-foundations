import copy

import torch 
import torch.optim as opt

from rlhf.evaluate import evaluate
from rlhf.sft import SFTTrainer
from utils.data_types import RLConfig
from utils.visualize import grad_norm


class AbstractRLHF:

    def __init__(self, config: RLConfig):
        self.config = config
    
    def prepare_data(self, batch_with_gens_and_rewards, tokenizer):
        """
        Returns the following:
        - input_ids with (batch * K) x max_seq 
        - att_mask with (batch * K) x max_seq
        - response_mask with (batch * K) x max_seq
        - advantage of size (batch * K)
        """
        
        content = []
        prompt_lens = []
        advantages = []

        for item, generations, rewards in batch_with_gens_and_rewards:
            assert len(generations) == self.config.K
            for generation in generations:
                prompt_len = len(tokenizer.tokenize(item["prompt"]))
                
                prompt_lens.append(prompt_len)
                content.append(item["prompt"] + generation)
            
            advantages.extend(rewards)
        
        inputs = tokenizer(content, padding=True, return_tensor="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Response mask is attention mask along with also masking out the prefix tokens
        response_mask = attention_mask.clone()
        for i, prompt_len in enumerate(prompt_lens):
            response_mask[i, :prompt_len] = 0.0
        
        advantages = torch.FloatTensor(advantages).view(-1, self.config.K)      # batch x K
        advantages = advantages - advantages.mean(dim=1, keepdim=True)

        if self.config.adv_normalize:
            std = advantages.std(dim=1, keepdim=True)
            advantages /= (std + 1e-6)
        
        advantages = advantages.view(-1)        # (batch * K)
                
        return input_ids, attention_mask, response_mask, advantages
    
    def calc_log_prob(self, model, input_ids, attention_mask, calc_entropy=False, noise=False):

        logits, _ = model(input_ids, attention_mask, noise=noise)  # (batch * K) x max_seq x vocab
        log_prob = torch.log_softmax(logits, dim=2)                # (batch * K) x max_seq x vocab

        entropy = None
        if calc_entropy:
            entropy = - (torch.exp(log_prob) * log_prob).sum(2)  # (batch * K) x max_seq

        log_prob = torch.gather(log_prob[:, :-1, :], 
                                index=input_ids[:, 1:].unsqueeze(2), 
                                dim=2)                  # (batch * K) x max_seq - 1 x 1
        log_prob = log_prob.squeeze(2)                  # (batch * K) x max_seq - 1

        return log_prob, entropy
        
    def calc_loss(self, log_prob, old_log_prob, infer_old_log_prob, response_mask, advantages):
        raise NotImplementedError()

    @torch.no_grad()
    def generate(self, model, prompts: list[str], tokenizer, K: int = 1) -> list[list[str]]:
        """Batched autoregressive sampling. Returns K completions per prompt."""
        
        if not prompts:
            return []

        temp = self.config.inference.temp
        max_tokens = self.config.inference.max_tokens

        device = next(model.parameters()).device
        pad_id = tokenizer.PAD_ID
        eos_id = tokenizer.eos
        model.eval()

        expanded_prompts = [prompt for prompt in prompts for _ in range(K)]
        inputs = tokenizer(expanded_prompts, padding=True, return_tensor="pt")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        batch_size = input_ids.size(0)
                
        # New tokens are appended after the padded prompt block; decode from here
        # (not from per-row prompt_lens, which would land on right-pad slots).
        gen_start = input_ids.size(1)

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

    def collect_data(self, data, model, tokenizer, reward_fn, 
                     K=None, visualizer=None, log_step: int | None = None, log_phase: str = "rl"):

        K = self.config.K if K is None else K
        prompts = data["prompt"]
        metadata_list = data["metadata"]

        all_generations = self.generate(model, prompts, tokenizer, K=K)

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

    @staticmethod
    def _clone_reference_model(model):
        model_ref = copy.deepcopy(model)
        for param in model_ref.parameters():
            param.requires_grad = False
        return model_ref

    def train(self, model, train_loader, eval_loader, tokenizer, reward_fn, visualizer=None):
        """
        :param model: transformer model
        :param train_loader: train loader. Each datapoint contains of a prompt and some metadata
        :param eval_loader: eval loader. Each datapoint contains of a prompt and some metadata
        :param tokenizer: tokenizer object
        :param reward_fn: a reward function that takes prompt, metadata, and completion and returns a reward score.
        """

        if self.config.sft.enabled:
            print("Starting SFT stage...")
            SFTTrainer(self.config.sft).train(
                model, train_loader, eval_loader, tokenizer, reward_fn, self, visualizer
            )
            print("SFT stage complete. Starting RL...")

        model_ref = self._clone_reference_model(model) if self.config.kl > 0.0 else None
        if model_ref is not None:
            model_ref.eval()

        optimizer = opt.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        device = next(model.parameters()).device
        it = 1

        for epoch in range(self.config.max_epochs):

            print(f"Training epoch: {epoch}")

            for data in train_loader:

                # Data has two keys prompt and metadata each with list of entries

                # Step 1: Generate responses
                batch_with_gens_and_rewards, mean_reward = self.collect_data(
                    data, model, tokenizer, reward_fn, visualizer=visualizer, log_step=it, log_phase="rl"
                )

                # Step 2: Prepare the data for training
                input_ids, attention_mask, response_mask, advantages = self.prepare_data(
                    batch_with_gens_and_rewards, tokenizer
                )
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                response_mask = response_mask.to(device)
                advantages = advantages.to(device)
                
                model.eval()    

                # TODO: Only compute this for TIS and IcePop
                infer_old_log_prob = None
                if self.config.algorithm in ["tis", "icepop"]:
                    with torch.no_grad():
                        # We add noise to simulate mismatch between inference and training engines
                        infer_old_log_prob, _ = self.calc_log_prob(model, input_ids, attention_mask, noise=True)
                        infer_old_log_prob = infer_old_log_prob.detach()

                with torch.no_grad():
                    old_log_prob, _ = self.calc_log_prob(model, input_ids, attention_mask)
                    old_log_prob = old_log_prob.detach()
                
                if self.config.kl > 0.0:
                    with torch.no_grad():
                        ref_log_prob, _ = self.calc_log_prob(model_ref, input_ids, attention_mask)
                        ref_log_prob = ref_log_prob.detach()

                # Step 3: Do RLHF training
                model.train()
                for _ in range(self.config.num_updates):

                    log_prob, entropy = self.calc_log_prob(model, input_ids, attention_mask, calc_entropy=True)
                    
                    rlhf_loss = self.calc_loss(log_prob, old_log_prob, infer_old_log_prob, response_mask[:, 1:], advantages)
                    avg_entropy = ((entropy * response_mask).sum(1) / (response_mask.sum(1) + 1e-6)).mean()
                    
                    kl_loss = 0.0
                    if self.config.kl > 0.0:
                        # Compute standard loss
                        kl_loss = ((log_prob - ref_log_prob) * response_mask[:, 1:]).sum(1).mean(0)
                    else:
                        kl_loss = 0.0
                    
                    loss = rlhf_loss + self.config.kl * kl_loss + getattr(model, "moe_aux_loss", 0.0)

                    optimizer.zero_grad()
                    loss.backward()
                    gn = grad_norm(model)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    optimizer.step()

                    if visualizer is not None:
                        visualizer.log_rl_step(
                            step=it,
                            total_loss=float(loss.item()),
                            rlhf_loss=float(rlhf_loss.item()),
                            kl_loss=float(kl_loss if isinstance(kl_loss, float) else kl_loss.item()),
                            mean_reward=float(mean_reward),
                            grad_norm_value=gn,
                            avg_entropy=float(avg_entropy.item()),
                        )
                    
                    print(f"Iteration {it}: Total Loss={loss:.4f}: RLHF loss={rlhf_loss:.4f}, KL loss={kl_loss:.4f}, Mean rewards {mean_reward:.2f}.")
                    it += 1

                    if it % 20 == 0:
                        evaluate(self, eval_loader, model, tokenizer, reward_fn, visualizer=visualizer, phase="rl", step=it)

