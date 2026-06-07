import torch

from rlhf2.llm.transformer import Transformer
from rlhf2.utils.data_types import LLMConfig


def test_transformer_forward_shapes():
    vocab_size = 8
    cfg = LLMConfig(num_layers=2, dim=16, num_head=2, head_dim=8, max_seq=32)
    model = Transformer(cfg, vocab_size=vocab_size)

    batch, seq_len = 3, 5
    input_ids = torch.randint(1, vocab_size, (batch, seq_len))
    attention_mask = torch.ones(batch, seq_len)

    logits, kv_caches = model(input_ids, attention_mask)

    assert logits.shape == (batch, seq_len, vocab_size)
    assert len(kv_caches) == cfg.num_layers
    # Each cached key spans the full sequence length.
    assert kv_caches[0][0].shape[1] == seq_len
