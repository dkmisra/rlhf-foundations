import torch

from llm.moe import MixtureOfExpert


def test_moe_forward_preserves_shape_and_returns_loss():
    dim = 32
    moe = MixtureOfExpert(num_experts=10, dim=dim, top_k=4)

    x = torch.randn(3, 10, dim)
    out, loss = moe(x, return_loss=True)

    assert out.shape == x.shape
    assert loss.shape == ()
    assert torch.isfinite(loss)
