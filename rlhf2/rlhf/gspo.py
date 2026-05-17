import torch 
from rl.grpo import GRPO


class GSPOConfig:

    # optimization
    max_epochs = 10
    lr = 1e-3
    weight_decay = 0.1
    grad_clip = 1.0

    # Core GRPO hyperparameters
    K = 8       # Number of generations per K
    eps_high = 0.00028
    eps_low = 0.0002
    kl = 0.0
    num_updates = 1

    # Dr. GRPO setting
    adv_normalize = False 
    length_normalize = False

    # Generation setup
    max_tokens = 40
    temp = 1.0
    top_p = 1.0


class GSPO(GRPO):

    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def calc_loss(self, log_prob, old_log_prob, response_mask, advantages):
        """
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
