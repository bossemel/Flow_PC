from cond_indep_test import copula_estimator
import torch


def test_copula_estimator():
    obs = 100
    x_uni = torch.FloatTensor(obs, 1).uniform_(0, 1)
    y_uni = torch.FloatTensor(obs, 1).uniform_(0, 1)
    cond_uni = torch.FloatTensor(obs, 5).uniform_(0, 1)

    copula, __ = copula_estimator(x_uni, y_uni, cond_uni, num_epochs=1)
    assert torch.min(copula) >= 0
    assert torch.max(copula) <= 1

# @Todo: write test for marginal estimator
# @Todo: write test with uniform distr for copula estimator

if __name__ == '__main__':
    test_copula_estimator()
