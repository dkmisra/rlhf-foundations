import torch 
import torch.nn as nn 


class MLP(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, x):
        out = self.net(x)
        x = x + out
        return x


class MixtureOfExpert(nn.Module):

    def __init__(self, num_experts, dim, top_k=8):
        super().__init__()

        self.experts = nn.ModuleList([MLP(dim) for _ in range(num_experts)])
        assert top_k <= len(self.experts), f"Cannot pick more experts ({top_k}) than there are experts ({len(self.experts)})"
        self.top_k = top_k
        self.dim = dim

        self.W_proj = nn.Linear(self.dim, len(self.experts))
    
    def forward(self, x, return_loss=False):
        """
            x is of the size batch (B) x max_seq (N) x dim (d)
        """
        
        batch = x.size(0)
        max_seq = x.size(1)
        dim = x.size(2)

        x = x.view(batch * max_seq, -1)
        weights = self.W_proj(x)        # (B * N) x num_experts

        # Pick top_k experts
        # top_k_weights and top_k_expert_id are of size (B * N) x top_k
        top_k_weights, top_k_expert_id = torch.topk(weights, k=self.top_k, dim=-1)

        if self.training:
            # Add noise to the logits 
            top_k_weights += torch.randn_like(top_k_weights) * 1.0 / float(len(self.experts))

        top_k_prob = torch.softmax(top_k_weights, dim=-1)       # (B * N) x top_k

        # For saving the output
        out = torch.zeros_like(x)                               # (B * N) x d

        # Next two are useful for load_balancing loss
        experts_num_tokens = []
        # Accumulate all router_prob here
        router_prob = torch.zeros(len(self.experts)).to(x.device)     # num_experts


        for i, expert in enumerate(self.experts):
            # Find all tokens that are mapped to this expert
            expert_present = (top_k_expert_id == i).any(dim=1)  # (B * N)

            # Number of tokens assigned to this expert
            num_tokens = expert_present.sum().item()
            experts_num_tokens.append(num_tokens)

            if num_tokens > 0:
                # At least one token goes to this expert
                item_indices = expert_present.nonzero()                                   # M_expert x 1
                expert_in = torch.gather(x, index=item_indices.expand([-1, dim]), dim=0)  # M_expert x d
                expert_out = expert(expert_in)                                            # M_expert x d

                # Compute weight assigned to this expert for every token
                expert_prob = top_k_prob.masked_fill(top_k_expert_id != i, 0.0).sum(1)    # B * N
                expert_prob = torch.gather(expert_prob.unsqueeze(1), 
                                             index=item_indices, dim=0)        # M_expert x 1

                router_prob[i] += expert_prob.sum()
                
                weighted_out = expert_out * expert_prob                        # M_expert x d

                # We need to put it back into out
                out.scatter_add_(src=weighted_out, index=item_indices.expand([-1, dim]), dim=0)  # (B * N) x d
        
        # Reshape and add residual connection
        out = out.view(batch, max_seq, dim)
        out = out + x.view(batch, max_seq, dim)
        
        if return_loss:
            # Fraction of tokens assigned to each expert
            experts_num_tokens = torch.FloatTensor(experts_num_tokens)      # num_experts
            experts_num_tokens = experts_num_tokens.to(router_prob.device)
            experts_num_tokens /= experts_num_tokens.sum()                  # num_experts

            # Accumulate all router_prob here
            router_prob /= float(batch * max_seq)                           # num_experts
            load_balancing_loss = len(self.experts) * (router_prob * experts_num_tokens).sum()

            return out, load_balancing_loss
        else:
            return out


def main():
    dim = 32
    moe = MixtureOfExpert(num_experts=10, dim=dim, top_k=4)

    x = torch.randn(3, 10, dim)
    out, loss = moe(x, return_loss=True)

    print(f"Out is {out.shape} and load balancing loss is {loss:.2f}")


if __name__ == "__main__":
    main()    
