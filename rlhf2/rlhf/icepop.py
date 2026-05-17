import torch 

from rl.grpo import GRPO


class IcePopConfig:

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

    # Icepop
    icepop_a = 0.5
    icepop_b = 2

    # Generation setup
    max_tokens = 40
    temp = 1.0
    top_p = 1.0


class IcePop(GRPO):

    def __init__(self, config):
        super().__init__()
        self.config = config
    
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