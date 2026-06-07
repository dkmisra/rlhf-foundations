import math
import torch 
import torch.nn as nn 

from rlhf2.llm.rope import Rope
from rlhf2.llm.sinpos_absolute import SinPosAbsoluteEmbedding
from rlhf2.utils.data_types import LLMConfig
from rlhf2.llm.moe import MixtureOfExpert
from rlhf2.llm.experts import MLP, SwishGLUMLP


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

    def __init__(self, llm_config: LLMConfig):
        super().__init__()

        dim = llm_config.dim
        self.attn = MultiHeadAttention(dim, llm_config.num_head, llm_config.head_dim, llm_config.causal)
        self._moe_load_balancing_coef = 0.0
        self._ffn_is_moe = False

        if llm_config.ffn_type == "mlp":
            self.ffn = MLP(dim)
        elif llm_config.ffn_type == "moe":
            from rlhf2.llm.moe import MixtureOfExpert

            moe_config = llm_config.moe
            if moe_config is None:
                raise ValueError("llm_config.moe is required when ffn_type is 'moe'")
            self.ffn = MixtureOfExpert(
                moe_config.num_experts, dim, top_k=moe_config.top_k
            )
            self._ffn_is_moe = True
            self._moe_load_balancing_coef = moe_config.load_balancing_coef
        else:
            raise ValueError(f"Invalid ffn_type: {llm_config.ffn_type!r}")

    def forward(self, x, attention_mask, kv_cache=None, rope=None):
        
        x, new_kv_cache = self.attn(x, attention_mask, kv_cache, rope)
        use_moe_loss = (
            self._ffn_is_moe
            and self._moe_load_balancing_coef > 0
            and self.training
        )
        if use_moe_loss:
            out, moe_loss = self.ffn(x, return_loss=True)
            x = out + x
            self._last_moe_loss = moe_loss
        else:
            out = self.ffn(x)
            x = out + x
            self._last_moe_loss = None
        return x, new_kv_cache


class Transformer(nn.Module):

    def __init__(self, llm_config: LLMConfig, vocab_size: int):
        super().__init__()

        self.llm_config = llm_config
        self.num_layers = llm_config.num_layers
        self._moe_load_balancing_coef = (
            llm_config.moe.load_balancing_coef
            if llm_config.moe is not None
            else 0.0
        )
        self.embed = nn.Embedding(vocab_size, llm_config.dim)

        if llm_config.pos_embed_type == "rope":
            self.rope = Rope(head_dim=llm_config.head_dim)
            self.pos_embedding = None
        elif llm_config.pos_embed_type == "absolute":
            self.pos_embedding = SinPosAbsoluteEmbedding(
                llm_config.max_seq, llm_config.dim
            )
            self.rope = None
        else:
            raise ValueError(
                f"Invalid position embedding type: {llm_config.pos_embed_type}"
            )

        self.layers = nn.ModuleList(
            [TransformerBlock(llm_config) for _ in range(llm_config.num_layers)]
        )

        self.W_proj = nn.Linear(llm_config.dim, vocab_size)
    
    def forward(self, input_ids, attention_mask, kv_caches=None, noise=False):
        """
            input_ids: long tensor of size (bach, seq_len)
            attention_mask: float tensor of size (batch, kv_len + seq_len)
            kv_caches: list of tuples of size num_layers. Each tuple contains the key and value tensors of the previous layers.
                       The key and value tensors are of size (batch, kv_len, num_head, head_dim).
                       If None, it is assumed to be empty.
            noise: If True, then adds random noise to each layer to simulate mismatch that occurs due to inference-training. 
                   This is purely for simulation purposes.

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
        moe_aux_loss: torch.Tensor | float = 0.0
        for i, layer in enumerate(self.layers):

            if noise:
                x += torch.randn_like(x) * self.llm_config.noise_scale

            x, new_kv_cache = layer(
                x,
                attention_mask,
                None if kv_caches is None else kv_caches[i],
                rope=self.rope,
            )
            new_kv_caches.append(new_kv_cache)
            last_moe_loss = getattr(layer, "_last_moe_loss", None)
            if last_moe_loss is not None:
                moe_aux_loss = moe_aux_loss + last_moe_loss

        coef = self._moe_load_balancing_coef
        if isinstance(moe_aux_loss, torch.Tensor) and coef > 0:
            self.moe_aux_loss = moe_aux_loss * coef
        else:
            self.moe_aux_loss = torch.zeros((), device=x.device)

        logits = self.W_proj(x)

        return logits, new_kv_caches
