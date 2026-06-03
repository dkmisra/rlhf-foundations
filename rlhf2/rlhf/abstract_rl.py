import torch 
import torch.optim as opt

from utils.data_types import RLConfig


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
    
    def calc_log_prob(self, model, input_ids, attention_mask):

        logits, _ = model(input_ids, attention_mask)       # (batch * K) x max_seq x vocab
        log_prob = torch.log_softmax(logits, dim=2)     # (batch * K) x max_seq x vocab
        log_prob = torch.gather(log_prob[:, :-1, :], 
                                index=input_ids[:, 1:].unsqueeze(2), 
                                dim=2)                  # (batch * K) x max_seq - 1 x 1
        log_prob = log_prob.squeeze(2)                  # (batch * K) x max_seq - 1
        return log_prob 
        
    def calc_loss(self, log_prob, old_log_prob, response_mask, advantages):

        ratio = torch.exp(log_prob - old_log_prob)     # (batch * K) x max_seq - 1
        clipped_term = torch.clamp(ratio, min=1 - self.config.eps_low, max=1 + self.config.eps_high) * advantages.unsqueeze(1)

        loss = - torch.minimum(ratio * advantages.unsqueeze(1), clipped_term)   # (batch * K) x max_seq - 1
        loss = loss * response_mask                    # (batch * K) x max_seq - 1
        loss = loss.sum(1)                             # (batch * K)

        response_len = response_mask.sum(1).float()  # (batch * K)    
        if self.config.length_normalize:
            loss /= (response_len + 1e-6)
        else:
            # Normalize each batch by total response
            response_len = response_len.view(-1, self.config.K)     # Batch x K
            response_len = response_len.sum(1, keepdim=True)      # batch x 1
            response_len = response_len.expand([-1, self.config.K]) # batch x K
            response_len = response_len.reshape(-1)
            loss /= (response_len + 1e-6)

        return loss.mean()

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
        prompt_lens = attention_mask.sum(dim=1).long()

        batch_size = input_ids.size(0)
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_tokens):
            logits, _ = model(input_ids, attention_mask)
            probs = torch.softmax(logits[:, -1, :] / max(temp, 1e-6), dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

            if eos_id is not None:
                done = done | (next_tokens == eos_id)
            done = done | (next_tokens == pad_id)
            next_tokens = torch.where(done, torch.full_like(next_tokens, pad_id), next_tokens)

            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(1)], dim=1)
            attention_mask = torch.cat([attention_mask, (~done).float().unsqueeze(1)], dim=1)

        completions: list[str] = []
        for i in range(batch_size):
            gen_ids: list[int] = []
            for token_id in input_ids[i, prompt_lens[i] :].tolist():
                if token_id == pad_id or (eos_id is not None and token_id == eos_id):
                    break
                gen_ids.append(token_id)
            completions.append(tokenizer.decode(gen_ids))

        return [completions[i * K : (i + 1) * K] for i in range(len(prompts))]

    def collect_data(self, data, model, tokenizer, reward_fn, K=None):

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

        mean_reward = sum_reward / float(len(prompts) * K)
        return batch_with_gens_and_rewards, mean_reward
    
    def evaluate(self, eval_loader, model, tokenizer, reward_fn):

        sum_rewards = 0.0
        num_samples = 0
        for data in eval_loader:
            _, mean_reward = self.collect_data(data, model, tokenizer, reward_fn, K=1)
            sum_rewards += mean_reward * len(data["prompt"])
            num_samples += len(data["prompt"])
        
        mean_reward = sum_rewards / float(num_samples)
        print(f"Evaluation: Mean reward is {mean_reward}")
    
    def train(self, model, model_ref, train_loader, eval_loader, tokenizer, reward_fn):
        """
        :param model: transformer model
        :param model_ref: reference model
        :param train_loader: train loader. Each datapoint contains of a prompt and some metadata
        :param eval_loader: eval loader. Each datapoint contains of a prompt and some metadata
        :param tokenizer: tokenizer object
        :param reward_fn: a reward function that takes prompt, metadata, and completion and returns a reward score.
        """

        optimizer = opt.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        it = 1

        for epoch in range(self.config.max_epochs):

            print(f"Training epoch: {epoch}")

            for data in train_loader:

                # Data has two keys prompt and metadata each with list of entries

                # Step 1: Generate responses
                batch_with_gens_and_rewards, mean_reward = self.collect_data(data, model, tokenizer, reward_fn)
                
                # Step 2: Prepare the data for training
                input_ids, attention_mask, response_mask, advantages = self.prepare_data(batch_with_gens_and_rewards, tokenizer)

                with torch.no_grad():
                    old_log_prob = self.calc_log_prob(model, input_ids, attention_mask)
                    old_log_prob = old_log_prob.detach()
                
                if self.config.kl > 0.0:
                    with torch.no_grad():
                        ref_log_prob = self.calc_log_prob(model_ref, input_ids, attention_mask)
                        ref_log_prob = ref_log_prob.detach()

                # Step 3: Do RLHF training
                for _ in range(self.config.num_updates):

                    log_prob = self.calc_log_prob(model, input_ids, attention_mask)
                    
                    grpo_loss = self.calc_loss(log_prob, old_log_prob, response_mask[:, 1:], advantages)
                    
                    kl_loss = 0.0
                    if self.config.kl > 0.0:
                        # Compute standard loss
                        kl_loss = ((log_prob - ref_log_prob) * response_mask[:, 1:]).sum(1).mean(0)
                    else:
                        kl_loss = 0.0
                    
                    loss = grpo_loss + self.config.kl * kl_loss

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    optimizer.step()
                    
                    print(f"Iteration {it}: Loss={loss:.4f}: GRPO loss={grpo_loss:.4f}, KL loss={kl_loss:.4f}, Mean rewards {mean_reward:.2f}.")
                    it += 1

                    if it % 20 == 0:
                        self.evaluate(eval_loader, model, tokenizer, reward_fn)

