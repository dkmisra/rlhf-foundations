import math
import torch 
import torch.nn as nn 

from llm.rope import Rope
from llm.sinpos_absolute import SinPosAbsoluteEmbedding


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


class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_head, head_dim, causal=True):
        super().__init__()
        self.dim = dim 
        self.num_head = num_head
        self.head_dim = head_dim
        self.causal = causal
        self.temp = math.sqrt(head_dim)

        self.normed_proj = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 3 * num_head * head_dim, bias=False)
        )
        self.W_out = nn.Linear(num_head * head_dim, dim, bias=False)
    
    def forward(self, x, attention_mask, kv_cache=None, rope=None):

        batch = x.size(0)
        seq_len = x.size(1)

        y = self.normed_proj(x)      # B x L x 3 Nh Hd
        y = y.view(batch, seq_len, self.num_head, 3 * self.head_dim)

        # k, q, v are each B x L x Nh x Hd
        k, q, v = torch.chunk(y, chunks=3, dim=-1)

        # Add KV cache, if exist
        if kv_cache is not None:
            k_old, v_old = kv_cache
            k = torch.cat([k_old, k], dim=1)
            v = torch.cat([v_old, v], dim=1)
        new_kv_cache = (k, v)

        if rope is not None:
            kv_len = 0 if kv_cache is None else kv_cache[0].shape[1]
            if attention_mask is None:
                rope_pos_ids = torch.arange(k.shape[1]).expand([batch, -1]).float().to(k.device)
            else:
                rope_pos_ids = torch.cumsum(attention_mask, dim=1) - 1
            k = rope(k, pos_ids=rope_pos_ids)
            q = rope(q, pos_ids=rope_pos_ids[:, kv_len:])

        k = k.permute(0, 2, 3, 1)       # B x Nh x Hd x L'
        q = q.permute(0, 2, 1, 3)       # B x Nh x L x Hd
        v = v.permute(0, 2, 1, 3)       # B x Nh x L' x Hd 

        # B x Nh x L x L'
        logits = q @ k / self.temp

        mask = None
        if self.causal:
            kv_len = 0 if kv_cache is None else kv_cache[0].shape[1]
            mask = torch.tril(torch.ones_like(logits), diagonal=kv_len).bool()      # B x Nh x L x L'
        
        if attention_mask is not None:
            # For something to be not ignored, it has to be both not in attention mask and causal (if causal set to True)
            attention_mask = attention_mask[:, None, None, :]       # B x 1 x 1 x L'
            mask = attention_mask.bool() if mask is None else mask & attention_mask.bool()

        if mask is not None:
            # Any place where mask is False/0, fill it with -inf
            logits = logits.masked_fill(~mask, -float('inf'))

        prob = torch.softmax(logits, dim=-1)    # B x Nh x L x L'
        value = prob @ v                        # B x Nh x L x Hd

        value = value.permute(0, 2, 1, 3).reshape(batch, seq_len, -1)  # B x L x (Nh Hd)
        value = self.W_out(value)       # B xL x d

        return value + x, new_kv_cache


class TransformerBlock(nn.Module):

    def __init__(self, dim, num_head, head_dim, causal=True):
        super().__init__()

        self.attn = MultiHeadAttention(dim, num_head, head_dim, causal)
        self.mlp = MLP(dim)
    
    def forward(self, x, attention_mask, kv_cache=None, rope=None):
        x, new_kv_cache = self.attn(x, attention_mask, kv_cache, rope)
        x = self.mlp(x)
        return x, new_kv_cache


class Transformer(nn.Module):

    def __init__(self, num_layers, vocab_size, dim, num_head, head_dim, max_seq, causal=True, pos_embed_type="rope"):
        super().__init__()

        self.num_layers = num_layers
        self.embed = nn.Embedding(vocab_size, dim)

        if pos_embed_type == "rope":
            self.rope = Rope(head_dim=head_dim)
            self.pos_embedding = None
        elif pos_embed_type == "absolute":
            self.pos_embedding = SinPosAbsoluteEmbedding(max_seq, dim)
            self.rope = None
        else:
            raise ValueError(f"Invalid position embedding type: {pos_embed_type}")

        self.layers = nn.ModuleList([
                TransformerBlock(dim, num_head, head_dim, causal) for _ in range(num_layers)
                ])
        
        # Final projection layer to get logits
        self.W_proj = nn.Linear(dim, vocab_size)
    
    def forward(self, input_ids, attention_mask, kv_caches=None):
        """
            input_ids: long tensor of size (bach, seq_len)
            attention_mask: float tensor of size (batch, kv_len + seq_len)
            kv_caches: list of tuples of size num_layers. Each tuple contains the key and value tensors of the previous layers.
                       The key and value tensors are of size (batch, kv_len, num_head, head_dim).
                       If None, it is assumed to be empty.

            Returns:
                logits: float tensor of size (batch, seq_len, vocab_size)
                new_kv_caches: list of tuples of size num_layers. Each tuple contains the key and value tensors of the current layers.
                                The key and value tensors are of size (batch, kv_len + seq_len, num_head, head_dim).
        """

        x = self.embed(input_ids)       # (batch, seq_len, dim)

        if self.pos_embedding is not None:
            kvlen = 0 if kv_caches is None else kv_caches[0][0].shape[1]
            # We only need to add position embedding to the sequence part, not the KV cache part
            position_ids = (torch.cumsum(attention_mask, dim=1) - 1)[:, kvlen:].long()
            pos_embed = self.pos_embedding(position_ids)
            x += pos_embed

        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            x, new_kv_cache = layer(x, attention_mask, None if kv_caches is None else kv_caches[i], rope=self.rope)
            new_kv_caches.append(new_kv_cache)
        
        logits = self.W_proj(x)
        
        return logits, new_kv_caches
