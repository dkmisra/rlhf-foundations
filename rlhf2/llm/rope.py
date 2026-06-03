import torch 
import torch.nn as nn 


class Rope(nn.Module):

    def __init__(self, head_dim, base=10000):
        super().__init__()
        self.head_dim = head_dim 
        assert head_dim % 2 == 0

        # Create the theta vectors
        thetas = torch.pow(base, - 2 * torch.arange(head_dim / 2) / head_dim)

        # Register buffers moves these tensors to same devices as the module.
        self.register_buffer("thetas", thetas)

    def forward(self, x, pos_ids=None):
        """
            x is of size batch x seq_len x num_head x head_dim
            pos_ids is of size batch x seq_len and denotes the position ids of the tokens. If None, it is assumed to be 0:seq_len-1.
        """

        batch, seq_len, num_head, head_dim = x.shape

        if pos_ids is None:
            pos_ids = torch.arange(seq_len).expand([batch, -1]).float().to(x.device)  # batch x seq_len
        
        pos_theta = pos_ids.view(batch, seq_len, 1) @ self.thetas.view(1, -1)         # batch x seq_len x head_dim/2
        cos = torch.cos(pos_theta)                                                    # batch x seq_len x head_dim/2
        sin = torch.sin(pos_theta)                                                    # batch x seq_len x head_dim/2
        
        x1 = x[:, :, :, 0::2]       # B x N x H x Hd/2
        x2 = x[:, :, :, 1::2]       # B x N x H x Hd/2

        # Even cos
        x1cos = cos[:, :, None, :] * x1     # B x N x H x Hd/2

        # Odd cos
        x2cos = cos[:, :, None, :] * x2     # B x N x H x Hd/2

        # Even sin
        x2sin = - sin[:, :, None, :] * x2   # B x N x H x Hd/2

        # Odd sin
        x1sin = sin[:, :, None, :] * x1     # B x N x H x Hd/2

        even_terms = x1cos + x2sin  # B x N x H x Hd/2
        odd_terms = x2cos + x1sin   # B x N x H x Hd/2

        return torch.cat([even_terms.unsqueeze(-1), odd_terms.unsqueeze(-1)], dim=-1).view(batch, seq_len, num_head, head_dim)