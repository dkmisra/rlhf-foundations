"""
Utilities for computing KL divergence
"""

import torch


def calc_k1(logp, logq, mask):
    """
        logp: A batch x N float tensor denoting logp for a sequence with logp[i] = log P(y_i | y_{<i})
        logp: A batch x N float tensor denoting logp for a sequence with logp[i] = log P(y_i | y_{<i})
        mask: A 0-1 batch x N float mask tensor where 1 means valid and 0 means invalid (e.g., padding)

        return: K1 sampling for KL divergence D_KL(p||q).

        K1 estimator is D_KL(p||q) = E_p[log p/q] ~ log p/q assuming data is sampling according to p. 
        It is unbiased but high-variance and potentially negative.
    """

    num_tokens = mask.sum().float().clamp(min=1.0)
    per_token_term = logp - logq
    return (per_token_term * mask).sum(1).mean(0) / num_tokens


def calc_k2(logp, logq, mask):
    """
        logp: A batch x N float tensor denoting logp for a sequence with logp[i] = log P(y_i | y_{<i})
        logp: A batch x N float tensor denoting logp for a sequence with logp[i] = log P(y_i | y_{<i})
        mask: A 0-1 batch x N float mask tensor where 1 means valid and 0 means invalid (e.g., padding)

        return: K2 sampling for KL divergence D_KL(p||q).

        K2 estimator is D_KL(p||q) = E_p[log p/q] = (log p - logq)^2 assuming data is sampling according to p. 
        It is biased, positive but high-variance.
        
        The derivation follows from Taylor series of log(1+x) = x - x^2/2 + x^3/3 - x^4/4 ..... We will use 
        2nd order approximation so log(1+x) ~ x - x^2/2.

        E_p[logp/q] = - E_p[logq/p] = -E[log(1 + q/p - 1)] = - E_p[log(1 + r)] where r = (q/p-1)
        then using 2nd order expansion we get

        - E_p[log(1+r)] ~ -E_p[r - r^2/2] = E_p[r] + E_p[r^2/2]. We have E_p[r] = E_p[q/p - 1] = \sum_i {p(i)q(i)}/p(i) - 1 = 0.

        Therefore, the estimate is given by E_p[r^2/2] ~ r^2/2 = 1/2(q/p -1 )^2 which is the K2 estimation.
    """

    num_tokens = mask.sum().float().clamp(min=1.0)

    # r term in the comment
    r = torch.exp(logq - logp)  
    per_token_term = 0.5 * (r - 1) **2

    return (per_token_term * mask).sum(1).mean(0) / num_tokens


def calc_k3(logp, logq, mask):
    """
        logp: A batch x N float tensor denoting logp for a sequence with logp[i] = log P(y_i | y_{<i})
        logp: A batch x N float tensor denoting logp for a sequence with logp[i] = log P(y_i | y_{<i})
        mask: A 0-1 batch x N float mask tensor where 1 means valid and 0 means invalid (e.g., padding)

        return: K3 sampling for KL divergence D_KL(p||q). 

        K3 sampling is given by D_KL(p||q) = E_p[logp/q] = E_p[q/p - 1 + logp/q] where it is easy to see E_p[q/p - 1] = 0.

        K3 estimator is just D_KL(p||q) = E_p[logp/q] = E_p[q/p - 1 + logp/q] ~ q/p - 1 + logp/q.

        This estimator is unbiased and always positive. The positivity can be shown as follows:

        Let x = p/q then estimator is f(x) = 1/x - 1 + logx where x > 0. Gradient f'(x) = -1/x^2 + 1/x = (x-1)/x^2.
        
        Then f'(x) >= 0 for x>=1. We have f(1) = 1 - 1 + log1 = 0 and so f(x) >= 0 for all x >= 1. 

        For 0 < x < 1, we prove positivity as follows:

        Define t = 1/x, then for 0 < x < 1, we have t > 1. We have g(t) = t - 1 - logt. 
        
        This gives g(1) = 0 and g'(t) = 1 - 1/t = (t-1)/t. For t > 1, we have g'(t) > 0, therefore, g(t) > 0 for all t > 0.

        This means f(x) > 0 for 0 < x < 1. Taken together, f(x) > 0 for x > 0.
    """

    num_tokens = mask.sum().float().clamp(min=1.0)

    r = torch.exp(logq - logp)  
    per_token_term = r - 1 + logp - logq

    return (per_token_term * mask).sum(1).mean(0) / num_tokens


def compute_kl(logp, logq, mask, kl_estimator="k3"):

    if kl_estimator == "k1":
        return calc_k1(logp, logq, mask)

    elif kl_estimator == "k2":
        return calc_k2(logp, logq, mask)

    elif kl_estimator == "k3":
        return calc_k3(logp, logq, mask)

    else:
        raise AssertionError(f"KL estimator must be of type {{k1, k2, k3}}. Given kl={kl_estimator}.")
