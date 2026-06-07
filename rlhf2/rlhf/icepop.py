import torch

from rlhf2.rlhf.grpo import GRPO
from rlhf2.utils.data_types import RLConfig


class IcePop(GRPO):

    def __init__(self, config: RLConfig):
        super().__init__(config)

    def calc_loss(self, log_prob, old_log_prob, infer_old_log_prob, response_mask, advantages):

        ratio = torch.exp(log_prob - old_log_prob)     # (batch * K) x max_seq - 1
        clipped_term = torch.clamp(ratio, min=1 - self.config.eps_low, max=1 + self.config.eps_high) * advantages.unsqueeze(1)

        imp_samp = torch.exp(old_log_prob - infer_old_log_prob).detach()    # (batch * K) x max_seq - 1
        imp_samp_mask = (imp_samp < self.config.icepop_a) | (self.config.icepop_b < imp_samp)
        imp_samp = imp_samp.masked_fill(imp_samp_mask, 0.0)                 # (batch * K) x max_seq - 1

        loss = - imp_samp * torch.minimum(ratio * advantages.unsqueeze(1), clipped_term)   # (batch * K) x max_seq - 1
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
