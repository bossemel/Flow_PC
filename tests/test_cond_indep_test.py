from cond_indep_test import copula_estimator, hypothesis_test, mi_estimator
import torch
import numpy as np
from options import TrainOptions
from utils import create_folders
import os 
import json 
import random 


def test_copula_estimator():
    # Training settings
    args = TrainOptions().parse()   # get training options
    args.exp_name = 'testing'
    
    # Create Folders
    create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    obs = 100
    x_uni = torch.FloatTensor(obs, 1).uniform_(0, 1)
    y_uni = torch.FloatTensor(obs, 1).uniform_(0, 1)
    cond_uni = torch.FloatTensor(obs, 5).uniform_(0, 1)
    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")

    cop_flow = copula_estimator(x_uni, y_uni, cond_uni, num_epochs=1,  exp_name='testing', device=device)

# # @Todo: write test for marginal estimator
# # @Todo: write test with uniform distr for copula estimator

def test_hypothesis_test_1():
    n = 100
    error = np.random.normal(loc=0, scale=0.00000001, size=(n,1))
    mi_1 = np.zeros((n, 1)) + error
    assert hypothesis_test(mi_1) == True

def test_hypothesis_test_2():
    n = 100
    error = np.random.normal(loc=0, scale=0.00000001, size=(n,1))
    mi_1 = np.ones((n, 1)) + error
    assert hypothesis_test(mi_1) == False

def test_hypothesis_test_3():
    n = 100
    mi_1 = np.random.normal(loc=0, scale=0.00000001, size=(n,1))
    assert hypothesis_test(mi_1) == True


def test_hypothesis_test_4():
    n = 100
    mi_1 = np.random.normal(loc=1, scale=0.00000001, size=(n,1))
    assert hypothesis_test(mi_1) == False

def test_hypothesis_test_5():
    n = 100
    mi_1 = np.random.normal(loc=1, scale=2, size=(n,1))
    assert hypothesis_test(mi_1) == False



# def test_mi_estimator_independent():
#     # @Todo: create a class with a _sample function that outputs some copula values for either a conditionally independent or conditionally independent samples
#     sampler = Fixed_sampler(dependent=False)
#     use_cuda = True
#     device = torch.device("cuda:0" if use_cuda else "cpu")

#     # cop_samples = sampler.sample(num_samples=100) # @Todo: make this run in samples
#     # log_density = torch.empty((cop_samples.shape[0], cop_samples.shape[1]))
#     # for mm in range(1000):
#     #     log_density_ = sampler.log_prob(cop_samples[:, mm, :]).mean()
#     #     assert not torch.isnan(log_density_.sum())
#     #     log_density[:, mm] = sampler.log_prob(cop_samples[:, mm, :]) # @Todo: triple check if this is correct
#     # print('mean log:', torch.mean(log_density))
#     # print(' mean:', torch.mean(torch.exp(log_density)))
#     # exit()
#     num_runs = 50

#     with torch.no_grad():
#         mi_runs = []
#         ii = 0
#         while ii < num_runs:
#             mi_estimate = mi_estimator(sampler, device=device)
#             if not np.isnan(mi_estimate):
#                 mi_runs.append(mi_estimate)
#                 ii += 1
#     assert np.mean(mi_runs) == 0


# def test_mi_estimator_dependent():
#     # @Todo: create a class with a _sample function that outputs some copula values for either a conditionally independent or conditionally independent samples
#     sampler = Fixed_sampler(dependent=True)
#     use_cuda = True
#     device = torch.device("cuda:0" if use_cuda else "cpu")
#     sampler._sample(num_samples=3)
#     mi = mi_estimator(sampler, device=device)
#     assert mi == 0

# def test_marginal_transform_1d():
#     raise NotImplementedError

# def test_marginal_transform():
#     raise NotImplementedError

# def test_copula_indep_test():
#     raise NotImplementedError

if __name__ == '__main__':

    # Set Seed
    random_seed = 432
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)

