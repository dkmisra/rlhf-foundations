import torch
import torch.optim as optim

from rlhf.evaluate import evaluate
from utils.data_types import SFTConfig
from utils.visualize import grad_norm


class SFTTrainer:

    def __init__(self, config: SFTConfig):
        self.config = config

    def prepare_data(self, data, tokenizer):
        content = []
        prompt_lens = []

        for prompt, metadata in zip(data["prompt"], data["metadata"]):
            completion = metadata["target_completion"] + tokenizer.eos_token
            prompt_lens.append(len(tokenizer.tokenize(prompt)))
            content.append(prompt + completion)

        inputs = tokenizer(content, padding=True, return_tensor="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        response_mask = attention_mask.clone()
        for i, prompt_len in enumerate(prompt_lens):
            response_mask[i, :prompt_len] = 0.0

        return input_ids, attention_mask, response_mask

    def calc_log_prob(self, model, input_ids, attention_mask):
        logits, _ = model(input_ids, attention_mask)
        log_prob = torch.log_softmax(logits, dim=2)
        log_prob = torch.gather(
            log_prob[:, :-1, :],
            index=input_ids[:, 1:].unsqueeze(2),
            dim=2,
        )
        return log_prob.squeeze(2)

    def calc_loss(self, log_prob, response_mask):
        mask = response_mask[:, 1:]
        loss = -(log_prob * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return loss.mean()

    def train(self, model, train_loader, eval_loader, tokenizer, reward_fn, rollout_trainer, visualizer=None):
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        device = next(model.parameters()).device
        model.train()
        it = 1

        for epoch in range(self.config.max_epochs):
            print(f"SFT epoch: {epoch}")

            for data in train_loader:
                input_ids, attention_mask, response_mask = self.prepare_data(data, tokenizer)
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                response_mask = response_mask.to(device)

                log_prob = self.calc_log_prob(model, input_ids, attention_mask)
                loss = self.calc_loss(log_prob, response_mask) + getattr(model, "moe_aux_loss", 0.0)

                optimizer.zero_grad()
                loss.backward()
                gn = grad_norm(model)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
                optimizer.step()

                if visualizer is not None:
                    visualizer.log_sft_step(float(loss.item()), gn, step=it)

                print(f"SFT iteration {it}: loss={loss:.4f}")
                it += 1

                if self.config.eval_every > 0 and it % self.config.eval_every == 0:
                    evaluate(
                        rollout_trainer,
                        eval_loader,
                        model,
                        tokenizer,
                        reward_fn,
                        label="SFT evaluation",
                        visualizer=visualizer,
                        phase="sft",
                        step=it,
                    )
