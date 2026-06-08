import copy

import torch 
import torch.optim as opt

from rlhf2.rlhf.evaluate import evaluate_reward
from rlhf2.rlhf.inference import batch_generate_with_rewards
from rlhf2.utils.data_types import RLStageConfig
from rlhf2.utils.visualize import grad_norm


class GRPOTrainer:
    """Group Relative Policy Optimization (GRPO).

    Serves as the base for the GRPO-family of RL trainers: data preparation and
    the RL training loop are shared here, while subclasses override
    :meth:`calc_loss` to change the policy-optimization objective.

    Reference: https://arxiv.org/pdf/2402.03300
    """

    def __init__(self, config: RLStageConfig):
        self.config = config

    def needs_infer_log_prob(self) -> bool:
        """Whether training needs a separate (noisy) inference-engine log-prob.

        Off-policy corrections (TIS, IcePop) override this to True.
        """
        return False
    
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

    @staticmethod
    def _clone_reference_model(model):
        model_ref = copy.deepcopy(model)
        for param in model_ref.parameters():
            param.requires_grad = False
        return model_ref

    def evaluate(self, model, eval_loader, tokenizer, reward_fn, visualizer=None, step=None):
        """Mean task reward over the eval loader (one generation per prompt)."""
        mean_reward = evaluate_reward(
            model,
            eval_loader,
            tokenizer,
            reward_fn,
            temp=self.config.inference.temp,
            max_tokens=self.config.inference.max_tokens,
            visualizer=visualizer,
            log_step=step,
        )
        print(f"Evaluation: Mean reward is {mean_reward:.4f}")
        if visualizer is not None and step is not None:
            visualizer.log_metrics(step, {"eval_reward": mean_reward}, throttle=False)
        return mean_reward

    def train(self, model, train_loader, eval_loader, tokenizer, reward_fn, visualizer=None):
        """
        :param model: transformer model
        :param train_loader: train loader. Each datapoint contains of a prompt and some metadata
        :param eval_loader: eval loader. Each datapoint contains of a prompt and some metadata
        :param tokenizer: tokenizer object
        :param reward_fn: a reward function that takes prompt, metadata, and completion and returns a reward score.
        """

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
                batch_with_gens_and_rewards, mean_reward = batch_generate_with_rewards(
                    data, model, tokenizer, reward_fn,
                    temp=self.config.inference.temp,
                    max_tokens=self.config.inference.max_tokens,
                    K=self.config.K,
                    visualizer=visualizer, log_step=it, log_phase="train",
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

                infer_old_log_prob = None
                if self.needs_infer_log_prob():
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
                        # Compute K3 loss
                        # K3: D(p||q) ~ logp/q + q/p - 1
                        diff = log_prob - ref_log_prob
                        kl_loss = diff + torch.exp(-diff) - 1
                        kl_loss = (kl_loss * response_mask[:, 1:]).sum(1).mean(0)
                    else:
                        kl_loss = 0.0
                    
                    loss = rlhf_loss + self.config.kl * kl_loss + getattr(model, "moe_aux_loss", 0.0)

                    optimizer.zero_grad()
                    loss.backward()
                    gn = grad_norm(model)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                    optimizer.step()

                    if visualizer is not None:
                        visualizer.log_metrics(
                            it,
                            {
                                "total_loss": float(loss.item()),
                                "rlhf_loss": float(rlhf_loss.item()),
                                "kl_loss": float(kl_loss if isinstance(kl_loss, float) else kl_loss.item()),
                                "mean_reward": float(mean_reward),
                                "grad_norm": gn,
                                "entropy": float(avg_entropy.item()),
                            },
                        )

                    print(f"Iteration {it}: Total Loss={loss:.4f}: RLHF loss={rlhf_loss:.4f}, KL loss={kl_loss:.4f}, Mean rewards {mean_reward:.2f}.")
                    it += 1

                    if self.config.eval_every > 0 and it % self.config.eval_every == 0:
                        self.evaluate(model, eval_loader, tokenizer, reward_fn, visualizer=visualizer, step=it)

