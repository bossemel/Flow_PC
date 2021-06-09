import torch.optim as optim
import torch
import os
from pathlib import Path
from nflows import flows
import torch.nn as nn
from sklearn.model_selection import train_test_split
from data_provider import DataProvider
from torch.utils.data import DataLoader
import numpy as np 
import warnings
eps = 1e-10

def set_optimizer_scheduler(model, lr, weight_decay, amsgrad, epochs):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler

def nll_error(log_density):
    return -torch.mean(log_density)

def create_folders(args):
    args.exp_path = os.path.join('results', args.exp_name)
    args.figures_path = os.path.join(args.exp_path, 'figures')
    args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    args.experiment_saved_models = os.path.join('saved_models', args.exp_name)
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_saved_models).mkdir(parents=True, exist_ok=True)



def split_data_marginal(inputs, batch_size, num_workers=12):
    # Transform into data object
    data_train, data_val = train_test_split(inputs, test_size=0.2)
    data_val, data_test = train_test_split(inputs, test_size=0.5)

    data_train = DataProvider(data_train)
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                                        # @Todo: take these arguments from somewhere else
    data_val = DataProvider(data_val)
    loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    data_test = DataProvider(data_test)
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                                        # @Todo: take these arguments from somewhere else
    return loader_train, loader_val, loader_test


def split_data_copula(inputs_cond, batch_size, num_workers):
    data_train, data_val = train_test_split(inputs_cond, test_size=0.20)
    data_val, data_test = train_test_split(data_val, test_size=0.50)

    data_train = DataProvider(inputs=data_train[:, :2], cond_inputs=data_train[:, 2:])
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                            # @Todo: take these arguments from somewhere else

    data_val = DataProvider(inputs=data_val[:, :2], cond_inputs=data_val[:, 2:])
    loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                            # @Todo: take these arguments from somewhere else

    data_test = DataProvider(inputs=data_test[:, :2], cond_inputs=data_test[:, 2:])
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                            # @Todo: take these arguments from somewhere else
    return loader_train, loader_val, loader_test


class Fixed_sampler(flows.Flow):
    def __init__(self, dependent=False):
        super().__init__(transform=None, distribution=None)
        self.dependent = dependent 

    def sample(self, num_samples, context=None):
        if not self.dependent:
            return torch.from_numpy(scipy.stats.norm.rvs(size=(num_samples, 2)))

    def log_prob(self, inputs, context=None):
        if not self.dependent:
            mult_norm = scipy.stats.multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
            return torch.log(torch.from_numpy(mult_norm.pdf(inputs)))

    #def _distribution(self):



def gaussian_change_of_var_ND(inputs: torch.Tensor, original_log_pdf, context=None):
    inputs[inputs == 0] = eps
    inputs[inputs == 1] = 1 - eps
    normal_distr = torch.distributions.normal.Normal(0, 1)
    assert torch.max(inputs) < 1, '{}'.format(torch.max(inputs))
    assert torch.min(inputs) > 0, '{}'.format(torch.min(inputs))

    recast_inputs = normal_distr.icdf(inputs).float()

    if context is not None:
        context[context == 0] = eps
        context[context == 1] = 1 - eps
        assert torch.max(context) < 1, '{}'.format(torch.max(context))
        assert torch.min(context) > 0, '{}'.format(torch.min(context))
        recast_context = normal_distr.icdf(context).float()
        copy_recast_inputs = recast_inputs.detach()
        original_joint = original_log_pdf(copy_recast_inputs, context=recast_context)
    else:
        copy_recast_inputs = recast_inputs.detach()
        original_joint = original_log_pdf(copy_recast_inputs)

    if context is not None:
        recast_inputs = torch.cat([recast_inputs, recast_context], axis=1)
    second_dim = recast_inputs.shape[1] if len(recast_inputs.shape) == 2 else 1

    if second_dim >= 2:
        determinant = normal_distr.log_prob(recast_inputs).sum(axis=1) # torch.exp(normal_distr.log_prob(recast_inputs).sum(axis=1))
    else:
        determinant = normal_distr.log_prob(recast_inputs) # torch.exp(normal_distr.log_prob(recast_inputs))

    output = original_joint - determinant
    #(output) >= 0, '{}'.format(torch.min(output))
    return output


def js_divergence(prob_x_in_p, prob_x_in_q,
                  prob_y_in_p, prob_y_in_q):
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