import torch 

from rl.grpo import GRPO


class CISPOConfig:

    # optimization
    max_epochs = 10
    lr = 1e-3
    weight_decay = 0.1
    grad_clip = 1.0

    # Core GRPO hyperparameters
    K = 8       # Number of generations per K
    eps_high = 0.28
    eps_low = 0.20
    kl = 0.0
    num_updates = 1

    # Dr. GRPO setting
    adv_normalize = False 
    length_normalize = False

    # Generation setup
    max_tokens = 40
    temp = 1.0
    top_p = 1.0


class CISPO(GRPO):

    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def calc_loss(self, log_prob, old_log_prob, response_mask, advantages):
        """
            CISPO objective is:

            J = 1/|batch| \sum_{batch} \sum_{i=1}^K \sum_{t=1}^|Y_i| clip(sgd(r_it), \rho) grad log p(y_it|x, y_{i, s<t}) A_i
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
