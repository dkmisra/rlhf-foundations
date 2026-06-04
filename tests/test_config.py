import pytest
from pydantic import ValidationError

from utils.data_types import LLMConfig, MoEConfig


def _base_kwargs(**overrides):
    kwargs = dict(num_layers=1, dim=16, num_head=2, head_dim=8, max_seq=32)
    kwargs.update(overrides)
    return kwargs


def test_moe_ffn_requires_moe_config():
    with pytest.raises(ValidationError):
        LLMConfig(**_base_kwargs(ffn_type="moe"))


def test_moe_top_k_cannot_exceed_num_experts():
    with pytest.raises(ValidationError):
        LLMConfig(
            **_base_kwargs(
                ffn_type="moe",
                moe=MoEConfig(num_experts=2, top_k=4),
            )
        )
