import torch

from rlhf.grpo import GRPO
from utils.data_types import RLConfig


class CISPO(GRPO):

    def __init__(self, config: RLConfig):
        super().__init__(config)

    def calc_loss(self, log_prob, old_log_prob, response_mask, advantages):
        r"""
            CISPO objective is:

            J = 1/|batch| \sum_{batch} \sum_{i=1}^K \sum_{t=1}^|Y_i| clip(sgd(r_it), \rho) grad log p(y_it|x, y_{i, <t}) A_i
        """

        ratio = torch.exp(log_prob - old_log_prob)                      # (batch * K) x max_seq - 1
        ratio = torch.clamp(ratio, 
                            min=1 - self.config.eps_low, 
                            max=1 + self.config.eps_high).detach()      # (batch * K) x max_seq - 1

        token_loss = - ratio * log_prob * advantages.unsqueeze(1)       # (batch * K) x max_seq - 1
        loss = (token_loss * response_mask).sum(1)                      # (batch * K)
        response_len = response_mask.sum(1).clamp(min=1.0)              # (batch * K)

        response_len = response_len.view(-1, self.config.K).sum(1)      # batch 
        loss = loss.view(-1, self.config.K)                             # batch x K

        loss = (loss / response_len[:, None]).sum(1).mean(0)
        return loss
