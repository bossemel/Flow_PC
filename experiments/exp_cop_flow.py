# @Todo: write test function that trains and evaluates conditional copula flow for conditional copula inputs
from cond_indep_test import copula_estimator
from options import TrainOptions
from utils import create_folders, HiddenPrints, gaussian_change_of_var_ND
import torch
import numpy as np
import json
import random
import os 
from utils.load_and_save import save_statistics
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF
from utils import create_folders, random_search, set_seeds
from data_provider import split_data_copula, Copula_Distr
from options import TrainOptions
from eval.plots import visualize_joint
from eval.metrics import jsd_copula
import statsmodels.api
eps = 1e-10


# @Todo: implement comparison (copula? kde?) e.g. statsmodels.nonparametric.kernel_density.KDEMultivariateConditional

def exp_cop_transform(inputs: torch.Tensor, copula_distr, cond_set_dim=None):
    # Transform into data object
    data_train, __, __, loader_train, loader_val, loader_test = split_data_copula(inputs[:, 0:1], 
                                                                                               inputs[:, 1:2], 
                                                                                               None, 
                                                                                               batch_size=128, 
                                                                                               num_workers=0, 
                                                                                               return_datasets=True)

    # Run experiment 
    experiment, __, test_metrics = copula_estimator(loader_train=loader_train, 
                                                                      loader_val=loader_val, 
                                                                      loader_test=loader_test, 
                                                                      cond_set_dim=None,
                                                                      exp_name=args.exp_name, 
                                                                      device=args.device, 
                                                                      amsgrad=args.amsgrad_c, 
                                                                      epochs=args.epochs_c, 
                                                                      num_workers=args.num_workers, 
                                                                      variable_num=0,
                                                                      n_layers=args.n_layers_c,
                                                                      hidden_units=args.hidden_units_c,
                                                                      tail_bound=args.tail_bound_c,
                                                                      lr=args.lr_c,
                                                                      weight_decay=args.weight_decay_c)

    # Plot results
    vizobs = 10000
    with torch.no_grad():
        samples = experiment.model.sample_copula(vizobs).cpu().numpy()
    visualize_joint(samples, experiment.figures_path, name=args.exp_name)

    # Calculate JSD # @Todo: figure out conditional inputs
    with torch.no_grad():
        jsd = jsd_copula(experiment.model, copula_distr, args.device, context=None, num_samples=100000)
    print(jsd)

    test_metrics['cop_flow_jsd'] = [jsd]
    experiment_logs = os.path.join('results', args.exp_name, 'cf', 'stats')

    # # Comparison to empirical CDF Transform:
    test_metrics['kde_jsd'] = kde_estimator(data_train)

    print('Flow JSD: {}, KDE JSD: {}'.format(test_metrics['cop_flow_jsd'][0], test_metrics['kde_jsd'][0]))
    save_statistics(experiment_logs, 'test_summary.csv', test_metrics, current_epoch=0, continue_from_mode=False)


def kde_estimator(data_train):
    kde_fit = scipy.stats.gaussian_kde(data_train.T)
    kde_fit = KDE_Decorator(kde_fit, args.device)
    return [jsd_copula(kde_fit, copula_distr, args.device, context=None, num_samples=100000)]

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


if __name__ == '__main__':
    # Training settings
    args = TrainOptions().parse()   # get training options
    args.exp_name = 'exp_cop_flow'
    # Generate the directory names
    args.flow_name = 'cf'

    # Create Folders
    create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.cuda.current_device()

    # Set Seed
    set_seeds(seed=args.seed, use_cuda=use_cuda)

    # Get inputs
    copula_distr = Copula_Distr(args.copula, theta=args.theta, transform=True)
    inputs = torch.from_numpy(copula_distr.sample(args.obs)) # @Todo: create conditional inputs
    
    # exp_cop_transform(inputs, copula_distr)
    # exit()

    # Transform into data object
    data_train, data_val, data_test, loader_train, loader_val, loader_test = split_data_copula(inputs[:, 0:1], 
                                                                                               inputs[:, 1:2], 
                                                                                               None, 
                                                                                               batch_size=128, 
                                                                                               num_workers=0, 
                                                                                               return_datasets=True)

    #random_search(loader_train, loader_val, loader_test, args.device, experiment_logs, iterations=200, epochs=50)
    random_search(copula_estimator, 'random_search_cop', loader_train, loader_val,
     loader_test, args.device, args.experiment_logs, iterations=200, epochs=args.epochs_c, flow_type='cop_flow')