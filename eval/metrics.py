import torch 
import warnings
from data_provider import Copula_Distr
eps = 1e-7


def jsd_copula(pred_dist, true_dist, device: str, num_samples: int =100000):
    """Returns JS-Divergence of the predicted distribution and the true distribution.
    Params:
        pred_dist: predicted distribution, copula flow
        true_dist: true distribution, copula distribution
        device: str, device to use
        num_samples: int, number of samples to use for Monte Carlo estimate
    Returns:    
        jsd: float, JS-Divergence
    """
    # Get ground truth
    samples_target = torch.from_numpy(true_dist.sample(num_samples=num_samples))
    samples_pred = pred_dist.sample_copula(num_samples=num_samples)

    # Prob X in both distributions
    prob_x_in_p = torch.exp(pred_dist.log_pdf_uniform(inputs=samples_pred)).to(device)
    prob_x_in_q = torch.from_numpy(true_dist.pdf(samples_pred.cpu())).to(device)

    # Prob Y in both distributions
    prob_y_in_p = torch.exp(pred_dist.log_pdf_uniform(inputs=samples_target.to(device))).to(device)
    prob_y_in_q = torch.from_numpy(true_dist.pdf(samples_target)).to(device)

    assert torch.min(prob_x_in_p) >= 0
    assert torch.min(prob_x_in_q) >= 0
    assert torch.min(prob_y_in_p) >= 0
    assert torch.min(prob_y_in_q) >= 0

    assert prob_x_in_p.shape == (num_samples,), '{}'.format(prob_x_in_p.shape)
    assert prob_x_in_q.shape == (num_samples,)
    assert prob_y_in_p.shape == (num_samples,)
    assert prob_y_in_q.shape == (num_samples,)

    divergence = js_divergence(prob_x_in_p=prob_x_in_p,
                                prob_x_in_q=prob_x_in_q,
                                prob_y_in_p=prob_y_in_p,
                                prob_y_in_q=prob_y_in_q)

    return divergence


def js_divergence(prob_x_in_p: torch.Tensor, prob_x_in_q: torch.Tensor,
                  prob_y_in_p: torch.Tensor, prob_y_in_q: torch.Tensor) -> float:
    """Calculate JS-Divergence using Monte Carlo.
    Params:
        prob_x_in_p: p(x), x from distr p(x), array
        prob_x_in_q: q(x), x from distr p(x), array
        prob_y_in_p: p(y), y from distr q(y), array
        prob_y_in_q: p(y), y from distr q(y), array
    Returns:
        divergence: int, JS-Divergence
    """
    assert prob_x_in_p.shape[0] == prob_x_in_q.shape[0]
    assert prob_x_in_q.shape[0] == prob_y_in_p.shape[0]
    assert prob_y_in_p.shape[0] == prob_y_in_q.shape[0]
    mix_X = prob_x_in_p + prob_x_in_q
    mix_Y = prob_y_in_p + prob_y_in_q

    prob_x_in_p[prob_x_in_p == 0] = 0 + eps
    prob_y_in_q[prob_y_in_q == 0] = 0 + eps

    assert torch.min(mix_X) > 0
    assert torch.min(mix_Y) > 0

    KL_PM = torch.log2((2 * prob_x_in_p) / mix_X)
    KL_PM[mix_X == 0] = 0
    KL_PM = KL_PM.mean()

    KL_QM = torch.log2((2 * prob_y_in_q) / mix_Y)
    KL_QM[mix_Y == 0] = 0
    KL_QM = KL_QM.mean()

    divergence = (KL_PM + KL_QM) / 2

    if divergence < 0 - 1e-05:
        warnings.warn("JSD estimate below zero. JSD: {}".format(divergence))

    return divergence

