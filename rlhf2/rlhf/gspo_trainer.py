import torch

from rlhf2.rlhf.grpo_trainer import GRPOTrainer
from rlhf2.utils.data_types import RLConfig


class GSPOTrainer(GRPOTrainer):
    """Group Sequence Policy Optimization (GSPO).

    Reference: https://arxiv.org/pdf/2507.18071
    """

    def __init__(self, config: RLConfig):
        super().__init__(config)

    def calc_loss(self, log_prob, old_log_prob, infer_old_log_prob, response_mask, advantages):
        r"""
            GSPO objective is:

            J = 1/|batch| \sum_{batch} \sum_{i=1}^K min{r A_i, clip(r, 1-e_low, 1 + e_high)A}
            r = (p(Y_i|x)/p_old(Y_i|x))^{1/|Y_i|}
              = exp(1/|Y_i| \sum_j log p(Y_ij|x, Y_{i,<j}) / p_old(Y_ij|x, Y_{i,<j}) )
        """

        ratio = (log_prob - old_log_prob) * response_mask       # (batch * K) x max_seq - 1
        ratio = ratio.sum(1)                                    # (batch * K)
        response_len = response_mask.sum(1)                     # (batch * K)
        ratio = torch.exp(ratio / (response_len + 1e-6))        # (batch * K)

        loss = - torch.minimum(ratio * advantages, 
                               torch.clamp(ratio, min=1 - self.config.eps_low, 
                                                  max=1 + self.config.eps_high) * advantages).mean()
        return loss
