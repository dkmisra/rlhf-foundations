import pytest
import torch

from rlhf.kl_utils import calc_k1, calc_k2, calc_k3, compute_kl


def _make_tensors():
    logp = torch.tensor([[0.0, -0.5, -1.0], [0.2, -0.3, -0.8]])
    logq = torch.tensor([[-0.2, -0.4, -0.9], [0.1, -0.5, -0.7]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    return logp, logq, mask


def test_kl_estimators_return_scalar_tensors():
    logp, logq, mask = _make_tensors()

    k1 = calc_k1(logp, logq, mask)
    k2 = calc_k2(logp, logq, mask)
    k3 = calc_k3(logp, logq, mask)

    assert k1.shape == torch.Size([])
    assert k2.shape == torch.Size([])
    assert k3.shape == torch.Size([])


def test_kl_estimators_are_zero_when_policies_match():
    logp = torch.tensor([[0.0, -0.5], [-0.2, -0.3]])
    logq = logp.clone()
    mask = torch.ones_like(logp)

    assert calc_k1(logp, logq, mask).item() == 0.0
    assert calc_k2(logp, logq, mask).item() == 0.0
    assert calc_k3(logp, logq, mask).item() == 0.0


def test_k2_and_k3_are_non_negative():
    logp, logq, mask = _make_tensors()

    assert calc_k2(logp, logq, mask).item() >= 0.0
    assert calc_k3(logp, logq, mask).item() >= 0.0


def test_compute_kl_dispatches_to_estimators():
    logp, logq, mask = _make_tensors()

    assert torch.allclose(compute_kl(logp, logq, mask, "k1"), calc_k1(logp, logq, mask))
    assert torch.allclose(compute_kl(logp, logq, mask, "k2"), calc_k2(logp, logq, mask))
    assert torch.allclose(compute_kl(logp, logq, mask, "k3"), calc_k3(logp, logq, mask))
    assert torch.allclose(compute_kl(logp, logq, mask), calc_k3(logp, logq, mask))


def test_compute_kl_rejects_unknown_estimator():
    logp, logq, mask = _make_tensors()

    with pytest.raises(AssertionError, match="KL estimator must be of type"):
        compute_kl(logp, logq, mask, kl_estimator="k4")
