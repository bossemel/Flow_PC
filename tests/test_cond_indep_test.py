from cond_indep_test import copula_estimator, hypothesis_test, mi_estimator
import torch
import numpy as np
from options import TrainOptions
from utils import create_folders
from eval.plots import visualize_joint
import os 
import json 
import random 
from data_provider import split_data_copula, Copula_Distr
eps = 1e-10
# @Todo: write test that trains cond indep test for different inputs and evalute the results 
# @Todo: write test that evaluted the conditional mutual information function for different given copula functions 

def test_copula_estimator():
    # Training settings
    args = TrainOptions().parse()   # get training options
    args.exp_name = 'testing'
    args.flow_name = 'cf'

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

    kwargs = {'n_layers': args.n_layers_c,
              'lr': args.lr_c,
              'weight_decay': args.weight_decay_c,
              'amsgrad': args.amsgrad_c,
              'hidden_units': args.hidden_units_c,
              'tail_bound': args.tail_bound_c,
              'n_blocks': args.n_blocks_c,
              'dropout': args.dropout_c,
              'use_batch_norm': args.batch_norm_c,
              'tails': args.tails_c, 
              'n_bins': args.n_bins_c,
              'unconditional_transform': args.unconditional_transform_c}

    # Transform into data object
    data_train, data_val, data_test, loader_train, loader_val, loader_test = split_data_copula(x_uni, 
                                                                                               y_uni, 
                                                                                               cond_uni, 
                                                                                               batch_size=128, 
                                                                                               num_workers=0, 
                                                                                               return_datasets=True)
    for batch in loader_train:
        print(batch[0].shape, batch[1].shape)
        
    cop_flow = copula_estimator(loader_train, 
                                loader_val, 
                                loader_test, 
                                cond_set_dim=cond_uni.shape[-1], 
                                epochs=1, 
                                exp_name='testing', 
                                device=device, 
                                **kwargs)

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


class Distr_Wrapper():
    def __init__(self, distribution):
        self.distribution = distribution 
    
    def sample_copula(self, num_samples, context=None):
        return torch.from_numpy(self.distribution.sample(num_samples))

    def log_pdf_uniform(self, inputs, context=None):
        return torch.log(torch.from_numpy(self.distribution.pdf(inputs)))


def test_mi_estimator_independent():
   # Get inputs
    copula_distr = Copula_Distr('independent', theta=0+eps, transform=False)
    visualize_joint(copula_distr.sample(1000000), os.path.join('results', 'testing'), name='copula_indep')
    copula_distr = Distr_Wrapper(copula_distr)
    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    mi = mi_estimator(copula_distr, device=device, obs_n=100000, obs_m=1000)
    print('Clayton independent:', mi)
    assert np.isclose(mi, 0, atol=1e-02)


def test_mi_estimator_dependent_clayton():
   # Get inputs
    copula_distr = Copula_Distr('clayton', theta=0+eps, transform=False)
    visualize_joint(copula_distr.sample(1000000), os.path.join('results', 'testing'), name='copula_clayton_indep')
    copula_distr = Distr_Wrapper(copula_distr)
    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    mi = mi_estimator(copula_distr, device=device, obs_n=100000, obs_m=1000)
    print('Clayton independent:', mi)
    assert np.isclose(mi, 0, atol=1e-02)

   # Get inputs
    copula_distr = Copula_Distr('clayton', theta=2+eps, transform=False)
    visualize_joint(copula_distr.sample(1000000), os.path.join('results', 'testing'), name='copula_clayton_dep')
    copula_distr = Distr_Wrapper(copula_distr)
    
    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    mi = mi_estimator(copula_distr, device=device, obs_n=100000, obs_m=1000)
    print('Clayton dependent', mi)
    assert not np.isclose(mi, 0, atol=1e-02)



def test_mi_estimator_dependent_frank():
   # Get inputs
    copula_distr = Copula_Distr('frank', theta=eps, transform=False)
    visualize_joint(copula_distr.sample(1000000), os.path.join('results', 'testing'), name='copula_frank_indep')
    copula_distr = Distr_Wrapper(copula_distr)

    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    mi = mi_estimator(copula_distr, device=device, obs_n=100000, obs_m=1000)
    print('Frank independent', mi)
    assert np.isclose(mi, 0, atol=1e-02)

   # Get inputs
    copula_distr = Copula_Distr('frank', theta=20, transform=False)
    visualize_joint(copula_distr.sample(1000000), os.path.join('results', 'testing'), name='copula_frank_dep')
    copula_distr = Distr_Wrapper(copula_distr)

    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    mi = mi_estimator(copula_distr, device=device, obs_n=100000, obs_m=1000)
    print('Frank dependent', mi)
    assert not np.isclose(mi, 0, atol=1e-02)


def test_mi_estimator_dependent_gumbel():
   # Get inputs
    copula_distr = Copula_Distr('gumbel', theta=1+eps, transform=False)
    visualize_joint(copula_distr.sample(1000000), os.path.join('results', 'testing'), name='copula_gumbel_indep')
    copula_distr = Distr_Wrapper(copula_distr)

    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    mi = mi_estimator(copula_distr, device=device, obs_n=100000, obs_m=1000)
    print('Gumbel independent', mi)
    assert np.isclose(mi, 0, atol=1e-02)

   # Get inputs
    copula_distr = Copula_Distr('gumbel', theta=5+eps, transform=False)
    visualize_joint(copula_distr.sample(1000000), os.path.join('results', 'testing'), name='copula_gumbel_dep')
    copula_distr = Distr_Wrapper(copula_distr)

    use_cuda = True
    device = torch.device("cuda:0" if use_cuda else "cpu")
    mi = mi_estimator(copula_distr, device=device, obs_n=100000, obs_m=1000)
    print('Gumbel dependent', mi)
    assert not np.isclose(mi, 0, atol=1e-02)

# def test_marginal_transform_1d():
#     raise NotImplementedError

# def test_marginal_transform():
#     raise NotImplementedError

# def test_copula_indep_test():
#     raise NotImplementedError

if __name__ == '__main__':
    test_mi_estimator_independent()
    test_mi_estimator_dependent_clayton()
    test_mi_estimator_dependent_frank()
    test_mi_estimator_dependent_gumbel()