import torch 


class RewardModeling:

    def __init__(self):
        pass

    def get_reward_score(self, model, input_ids, attention_mask, response_mask):

        scores = model(input_ids, attention_mask)       # B x N

        # We use the last token with response_mask=1 for reward
        total_sum = response_mask.sum(1)
        cum_sum = torch.cumsum(total_sum, dim=1)
        mask = (cum_sum == total_sum) & (response_mask == 1)  # B x N

        reward = scores.masked_fill(mask == 0, 0.0).sum(1)
        return reward

    def calc_loss(self, batch_input, batch_mask, response_mask, model):
        """
            Computes the reward modeling loss.
                batch_input: 2b x n where batch_input[i] and batch_input[2i + 1] are pair of preferred and rejected smaples for all 0 < i <= b - 1
                batch_mask: 2b x n where batch_mask[i] and batch_mask[2i + 1] are pair of preferred and rejected samples for all 0 < i <= b - 1
                response_mask: 2b x n response_mask which is 1 where response_tokens are
        """

        scores = self.get_reward_score(model, batch_input, batch_mask, response_mask)
        batch_size = scores.size(0) // 2

        scores_win = scores[:batch_size]
        scores_loss = scores[batch_size:]
        
        loss = - torch.log_sigmoid(scores_win - scores_loss).mean()
        return loss