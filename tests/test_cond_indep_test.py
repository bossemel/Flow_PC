from cond_indep_test import copula_estimator
import numpy as np
import torch


def test_copula_estimator():
    obs = 100
    x_uni = torch.FloatTensor(obs, 1).uniform_(0, 1)
    y_uni = torch.FloatTensor(obs, 1).uniform_(0, 1)
    cond_uni = torch.FloatTensor(obs, 5).uniform_(0, 1)

    copula = copula_estimator(x_uni, y_uni, cond_uni, num_epochs=1)
    assert torch.min(copula) >= 0
    assert torch.max(copula) <= 1

if __name__ == '__main__':
    test_cond_indep_test()
