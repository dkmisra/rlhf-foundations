import torch 
import torch.optim as opt


class GRPOConfig:

    # optimization
    max_epochs = 10
    lr = 1e-3
    weight_decay = 0.1
    grad_clip = 1.0

    # Core GRPO hyperparameters
    K = 8       # Number of generations per K
    eps_high = 0.28
    eps_low = 0.2
    kl = 0.0
    num_updates = 1

    # Dr. GRPO setting
    adv_normalize = False 
    length_normalize = False

    # Generation setup
    max_tokens = 40
    temp = 1.0
    top_p = 1.0


class AbstractRLHF:

    def __init__(self, config):
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
                prompt_len = len(tokenizer(item["prompt"]))
                
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

    def collect_data(self, data, model, endpoint, reward_fn, K=None):

        K = self.config.K if K is None else K

        sum_reward = 0
        batch_with_gens_and_rewards = []
        for prompt, metadata in zip(data["prompt"], data["metadata"]):
            generations = []
            rewards = []
            for _ in range(K):
                generation = endpoint.generate(model, prompt, 
                                                temp=self.config.temp, max_tokens=self.config.max_tokens, top_p=self.config.top_p)
                reward = reward_fn(prompt, metadata, generation) 

                generations.append(generation)
                rewards.append(reward)
            sum_reward += sum(rewards)
            batch_with_gens_and_rewards.append(({"prompt": prompt, "metadata": metadata}, generations, rewards))

        mean_reward = sum_reward / float(len(batch_with_gens_and_rewards) * self.config.K)
        return batch_with_gens_and_rewards, mean_reward
    
    def evaluate(self, eval_loader, model, endpoint, reward_fn):

        sum_rewards = 0.0
        num_samples = 0
        for data in eval_loader:
            _, mean_reward = self.collect_data(data, model, endpoint, reward_fn, K=1)
            sum_rewards += mean_reward * len(data)  # TODO check len(data)
            num_samples += len(data)
        
        mean_reward = sum_rewards / float(num_samples)
        print(f"Evaluation: Mean reward is {mean_reward}")
    
    def train(self, model, model_ref, train_loader, eval_loader, endpoint, tokenizer, reward_fn):
        """
        :param model: transformer model
        :param model_ref: reference model
        :param train_loader: train loader. Each datapoint contains of a prompt and some metadata
        :param eval_loader: eval loader. Each datapoint contains of a prompt and some metadata
        :param endpoint: an endpoint with a function generate that takes a model and generation hyperparameters and returns a generation
        :param tokenizer: tokenizer object
        :param reward_fn: a reward function that takes prompt, generation and metadata and returns a reward score.
        """

        optimizer = opt.AdamW(model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        it = 1

        for epoch in range(self.config.max_epochs):

            print(f"Training epoch: {epoch}")

            for data in train_loader:

                # Data has two keys prompt and metadata each with list of entries

                # Step 1: Generate responses
                batch_with_gens_and_rewards, mean_reward = self.collect_data(data, model, endpoint, reward_fn)
                
                # Step 2: Prepare the data for training
                input_ids, attention_mask, response_mask, advantages = self.prepare_data(batch_with_gens_and_rewards, tokenizer)

                with torch.no_grad():
                    old_log_prob = self.calc_log_prob(model, input_ids, attention_mask)
                    old_log_prob = old_log_prob.detach()
                
                if self.config.kl > 0.0:
                    with torch.no_grad():
                        ref_log_prob = self.calc_log_prob(model_ref, input_ids, attention_mask)
                        ref_log_prob = ref_log_prob.detach()

                # Step 3: Do GRPO training
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
                        self.evaluate(eval_loader, model, endpoint, reward_fn)

