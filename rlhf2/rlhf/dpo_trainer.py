import torch 
import torch.nn.functional as F


class DPOTrainer:

    def __init__(self, beta, alpha=0.0, p=1.0):
        """
            :param beta: DPO hyperparameter controlling implicit KL
            :param alpha: length penalty
            :param p: for label smoothing. p=1.0 meanas no smoohing.
        """
        self.beta = beta 
        self.alpha = alpha 
        self.p = p 
    
    def get_log_prob(self, model, batch_input, attention_mask, response_mask):

        logits, _ = model(batch_input, attention_mask)  # logits is of size 2batch x max_seq x vocab
        log_prob = torch.log_softmax(logits, dim=2)     # 2batch x max_seq x vocab

        # of size 2batch x max_seq - 1
        selected_log_prob = torch.gather(log_prob[:, :-1], index=batch_input[:, 1:].unsqueeze(2), dim=2).squeeze(2) 
        log_prob = (selected_log_prob * attention_mask[:, 1:] * response_mask[:, 1:]).sum(1)   # 2batch
        return log_prob

    def calc_loss(self, batch_input, attention_mask, response_mask, model, model_ref):
        """
            batch_input, attention_mask, and response_mask are three tensors of size 2batch x max_seq where 
            (i, i + batch) are paired with accepted firs and rejected second for all i in {0, 1, 2, ..., batch -1}.

            :param batch_input: Long Tensor containing token ids
            :param attention_mask: Float Tensor containing 0-1 attention mask
            :param response_mask: Float Tensor containig 0-1 repsonse mask where 1 denotes a response token
            :param model: Transformer model
            :param model_ref: Reference transformer model
        """

        log_prob = self.get_log_prob(model, batch_input, attention_mask, response_mask)                # 2batch
        with torch.no_grad():
            ref_log_prob = self.get_log_prob(model_ref, batch_input, attention_mask, response_mask)    # 2batch
            ref_log_prob = ref_log_prob.detach()

        log_prob_win, log_prob_rej = log_prob.chunk(2, dim=0)          # each of size batch
        ref_log_prob_win, ref_log_prob_rej = ref_log_prob.chunk(2, dim=0)  # each of size batch

        term = self.beta * ((log_prob_win - ref_log_prob_win) - (log_prob_rej - ref_log_prob_rej))

        if self.alpha > 0.0:
            response_len = response_mask.sum(1)               # 2batch
            win_len, rej_len = response_len.chunk(2, dim=0)   # each of size batch
            length_penalty = win_len - rej_len
            term += self.alpha * length_penalty

        if self.p == 1.0:
            loss = - F.logsigmoid(term)                       # batch
        else:
            loss = - self.p * F.logsigmoid(term) - (1 - self.p) * F.logsigmoid(-term)   # batch

        loss = loss.mean()

        return loss
