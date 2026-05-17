import torch 
import torch.nn as nn 


class SinPosAbsoluteEmbedding(nn.Module):

    def __init__(self, max_seq, dim, base=10000):
        super().__init__()
        
        assert dim % 2 == 0
        # pos_{k, 2i} = sin(k/10000^{2i/d})
        # pos_{k, 2i+1} = cos(k/10000^{2i/d})

        pos = torch.arange(max_seq).float()
        frequencies = torch.pow(base, -2/float(dim) * torch.arange(dim//2)).float()

        # max_seq x dim/2
        thetak = pos.view(-1, 1) @ frequencies.view(1, -1)

        # max_seq x dim/2
        sin = torch.sin(thetak)
        cos = torch.cos(thetak)

        # max_seq x dim
        positional_embeddings = torch.cat([sin.unsqueeze(-1), cos.unsqueeze(-1)], dim=-1).view(max_seq, dim)

        self.pos_embeddings = nn.Embedding.from_pretrained(positional_embeddings, freeze=True)
    
    def forward(self, position_ids):
        """
            :param position_ids: tensor of position of size batch x seq_len
        """

        return self.pos_embeddings(position_ids)       # batch x seq_len x dim
