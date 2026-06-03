import torch
import torch.nn as nn 


class SwishGLUMLP(nn.Module):
    """
        Swish-GLU MLP implementation. See https://arxiv.org/pdf/2002.05202 for more details.
    """

    def __init__(self, dim, expansion_factor=4, beta=1):
        super().__init__()
        self.beta = beta
        self.norm = nn.LayerNorm(dim)
        self.W_up = nn.Linear(dim, expansion_factor * dim, bias=False)
        self.W_gate = nn.Linear(dim, expansion_factor * dim, bias=False)
        self.W_down = nn.Linear(expansion_factor * dim, dim)
    
    def forward(self, x):
        
        y = self.norm(x)
        up = self.W_up(y)
        gate_in = self.W_gate(y)
        if self.beta == 1:
            gate = torch.silu(gate_in)
        else:
            gate = gate_in * torch.sigmoid(self.beta * gate_in)
        return self.W_down(up * gate) + x


class MLP(nn.Module):
    """
        Basic MLP implementation.
    """

    def __init__(self, dim, expansion_factor=4):
        super().__init__()

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * expansion_factor),
            nn.SiLU(),
            nn.Linear(dim * expansion_factor, dim)
        )
    
    def forward(self, x):
        return x + self.net(x)
