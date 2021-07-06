import torch.optim as optim
import torch
import os
from pathlib import Path
import sys
import numpy as np
import random
import scipy
from eval.metrics import jsd_copula

eps = 1e-7


def set_optimizer_scheduler(model, lr, weight_decay, amsgrad, epochs):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler

def nll_error(log_density):
    return -torch.mean(log_density)

def create_folders(args):
    args.exp_path = os.path.join('results', args.exp_name, args.flow_name)
    args.figures_path = os.path.join(args.exp_path, 'figures')
    args.experiment_logs = os.path.join(args.exp_path, 'stats')
    args.experiment_saved_models = os.path.join('saved_models', args.exp_name)
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_saved_models).mkdir(parents=True, exist_ok=True)
    return args


class HiddenPrints:
    """Hide Prints (for grid search).
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# class Fixed_sampler(flows.Flow):
#     def __init__(self, dependent=False):
#         super().__init__(transform=None, distribution=None)
#         self.dependent = dependent 

#     def sample(self, num_samples, context=None):
#         if not self.dependent:
#             return torch.from_numpy(scipy.stats.norm.rvs(size=(num_samples, 2)))

#     def log_prob(self, inputs, context=None):
#         if not self.dependent:
#             mult_norm = scipy.stats.multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
#             return torch.log(torch.from_numpy(mult_norm.pdf(inputs)))

#     #def _distribution(self):



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


def random_search(estimator, flow_name, loader_train, loader_val, loader_test, device, 
                  experiment_logs, iterations, epochs, flow_type='marg_flow', cond_set_dim=None):
    results_dict = {}
    tested_combinations = []
    best_loss = 1000
    ii = 0
    while ii < iterations:
        n_layers = np.random.choice(range(1, 10))
        hidden_units = 2**np.random.choice(range(1, 7))
        n_bins = 5 * np.random.choice(range(2, 10))
        lr = 1 / 10**np.random.choice(range(2, 5))
        weight_decay = 1 / 10**(np.random.choice(range(2, 15)))
        tail_bound = 2**np.random.choice(range(5, 8)).item()
        amsgrad = np.random.choice([True, False])
        clip_grad_norm = np.random.choice([True, False])
        identity_init = np.random.choice([True, False])
        tails = np.random.choice(['linear', None])

        kwargs = {'n_layers': n_layers,
                    'hidden_units': hidden_units,
                    'n_bins': n_bins,
                    'lr': lr,
                    'weight_decay': weight_decay,
                    'tail_bound': tail_bound,
                    'amsgrad': amsgrad,
                    'clip_grad_norm': clip_grad_norm,
                    'identity_init': identity_init,
                    'tails': tails,
                    'cond_set_dim': cond_set_dim}
        if flow_type == 'cop_flow':
            dropout = 0.05 * np.random.choice(range(1, 6))
            use_batch_norm = np.random.choice([True, False])
            kwargs['dropout'] = dropout
            kwargs['use_batch_norm'] = use_batch_norm
            n_blocks = np.random.choice(range(1, 10))
            kwargs['n_blocks'] = n_blocks

            current_hyperparams = (n_layers,
                                   hidden_units,
                                   n_blocks,
                                   n_bins,
                                   dropout,
                                   lr,
                                   weight_decay,
                                   tail_bound,
                                   use_batch_norm,
                                   amsgrad,
                                   clip_grad_norm,
                                   identity_init,
                                   tails)

        elif flow_type == 'marg_flow':
            current_hyperparams = (n_layers,
                                   hidden_units,
                                   n_bins,
                                   lr,
                                   weight_decay,
                                   clip_grad_norm,
                                   tail_bound,
                                   identity_init,
                                   tails)

        else:
            raise ValueError('Unknown Flow type')

        if current_hyperparams not in tested_combinations:
            if flow_type == 'cop_flow':
                hyperparams_string = 'n_layers, hidden_units, n_blocks, n_bins, dropout, lr, weight_decay, \
                tail_bound, batch_norm, amsgrad, clip_grad, identity init, tails'
            else:
                hyperparams_string = 'n_layers, hidden_units, n_blocks, n_bins, lr, weight_decay, clip_grad_norm, \
                tail_bound, identity_init_m, tails_m'
            print('{}: {}'.format(hyperparams_string, current_hyperparams))
            with HiddenPrints():
                experiment, experiment_metrics, __ = estimator(loader_train=loader_train, 
                                                               loader_val=loader_val, 
                                                               loader_test=loader_test, 
                                                               exp_name='random_search', 
                                                               device=device, 
                                                               epochs=epochs, 
                                                               num_workers=0, 
                                                               variable_num=0,
                                                               disable_tqdm=True,
                                                               **kwargs)
            results_dict[current_hyperparams] = (experiment.best_val_model_idx, experiment.best_val_model_loss)
            with open(os.path.join(experiment_logs, 'random_search.txt'), 'w') as ff:
                ff.write(str(results_dict))
            if experiment.best_val_model_loss < best_loss:
                best_loss = experiment.best_val_model_loss
                best_epoch = experiment.best_val_model_idx
                best_hyperparams = current_hyperparams
            tested_combinations.append(current_hyperparams)
            ii += 1

    print('Random search complete for {}'.format(flow_name))
    print('Best hyperparams: {}'.format(best_hyperparams))
    print('Lowest Val Loss: {}'.format(best_loss))
    with open(os.path.join(experiment_logs, 'random_search.txt'), 'a') as ff:
        ff.write('Best hyperparams: {} Lowest Val Loss: {} Best Epoch: {}'.format(best_hyperparams, best_loss, best_epoch))


def set_seeds(seed, use_cuda=True):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)


def kde_estimator(data_train, copula_distr, device):
    kde_fit = scipy.stats.gaussian_kde(data_train.T)
    kde_fit = KDE_Decorator(kde_fit, device)
    return [jsd_copula(kde_fit, copula_distr, device, num_samples=100000)]

class KDE_Decorator():
    def __init__(self, model, device):
        self.model = model
        self.norm_distr = scipy.stats.norm()
        self.device = device

    def log_pdf_normal(self, inputs, context=None):
        return torch.log(torch.from_numpy(self.model.pdf(inputs.T.cpu().numpy()))).T.to(self.device)

    def log_pdf_uniform(self, inputs, context=None):
        #inputs = torch.from_numpy(self.norm_distr.ppf(inputs.cpu().numpy()))
        return gaussian_change_of_var_ND(inputs, self.log_pdf_normal, context=context).to(self.device)

    def sample_copula(self, num_samples, context=None):
        return torch.from_numpy(self.norm_distr.cdf(self.model.resample(num_samples).T)).to(self.device)
