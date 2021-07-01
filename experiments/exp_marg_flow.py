# @Todo: write test function that trains and evaluates conditional copula flow for conditional copula inputs
from cond_indep_test import marginal_estimator
from options import TrainOptions
from utils import create_folders, HiddenPrints
import torch
import numpy as np
import json
import random
import os 
from utils.load_and_save import save_statistics
import scipy.stats
from utils import create_folders, random_search, set_seeds
from data_provider import split_data_marginal, Marginal_Distr
from options import TrainOptions
from eval.plots import visualize1d
import matplotlib.pyplot as plt
eps = 1e-10


def kde_nll(data_train, data_test):
    norm_distr = scipy.stats.norm()
    kde = scipy.stats.gaussian_kde(data_train.reshape(-1,))
    cdf = np.vectorize(lambda x: kde.integrate_box_1d(-np.inf, x))
    uniform_samples = cdf(data_test.reshape(-1,))
    uniform_samples[uniform_samples == 0] = eps
    uniform_samples[uniform_samples == 1] = 1 - eps
    gaussian_samples = norm_distr.ppf(uniform_samples)
    return -np.mean(norm_distr.logpdf(gaussian_samples))

def exp_marg_transform(inputs):

    # Training settings
    args = TrainOptions().parse()   # get training options
    if args.exp_name == 'default_name':
        args.exp_name = 'exp_marg_flow'

    # Create Folders
    create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.cuda.current_device()

    # Set Seed
    set_seeds(args.seed)

    # Transform into data object
    data_train, data_val, data_test, loader_train, loader_val, loader_test = split_data_marginal(inputs, 
                                                                                                 batch_size=args.batch_size_m, 
                                                                                                 num_workers=0, 
                                                                                                 return_datasets=True)

    # Run experiment 
    experiment, experiment_metrics, test_metrics = marginal_estimator(loader_train=loader_train, 
                                                                      loader_val=loader_val, 
                                                                      loader_test=loader_test, 
                                                                      exp_name=args.exp_name, 
                                                                      device=args.device, 
                                                                      amsgrad=args.amsgrad_m, 
                                                                      epochs=args.epochs_m, 
                                                                      num_workers=args.num_workers, 
                                                                      variable_num=0,
                                                                      n_layers=args.n_layers_m,
                                                                      hidden_units=args.hidden_units_m,
                                                                      n_bins=args.n_bins_m,
                                                                      tail_bound=args.tail_bound_m,
                                                                      identity_init=args.identity_init_m,
                                                                      tails=args.tails_m,
                                                                      lr=args.lr_m,
                                                                      weight_decay=args.weight_decay_m)  # @Todo: make sure these are all actually used

    # Plot results
    visualize1d(experiment.model, device=args.device, path=experiment.figures_path, true_samples=data_train, obs=1000, name='marg_flow')

    experiment_logs = os.path.join('results', args.exp_name, 'mf_0', 'stats')

    # Comparison to empirical CDF Transform:
    ecdf_nll = kde_nll(data_train, data_test)
    test_metrics['ecdf_nll'] = [ecdf_nll]

    print('Flow NLL: {}, ECDF NLL: {}'.format(test_metrics['test_loss'][0], ecdf_nll))
    save_statistics(experiment_logs, 'test_summary.csv', test_metrics, current_epoch=0, continue_from_mode=False)


if __name__ == '__main__':
    # Training settings
    args = TrainOptions().parse()
    args.exp_name = 'exp_marg_flow'
    # Generate the directory names
    flow_name = 'mf_0'
    args.exp_path = os.path.join('results', args.exp_name, flow_name)
    args.figures_path = os.path.join(args.exp_path, 'figures')
    args.experiment_logs = os.path.join(args.exp_path, 'stats')
    args.experiment_saved_models = os.path.join('saved_models', args.exp_name)

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
    marginal_data = Marginal_Distr(args.marginal, mu=args.mu, var=args.var, alpha=args.alpha, low=args.low, high=args.high)
    inputs = marginal_data.sample(args.obs)
    
    exp_marg_transform(inputs)
    exit()
    

    # Transform into data object
    data_train, data_val, data_test, loader_train, loader_val, loader_test = split_data_marginal(inputs, 
                                                                                                 batch_size=args.batch_size_m, 
                                                                                                 num_workers=0, 
                                                                                                 return_datasets=True)

    random_search(marginal_estimator, 
                  'random_search_marg', 
                  loader_train, 
                  loader_val, 
                  loader_test, 
                  args.device, 
                  args.experiment_logs, 
                  iterations=200, 
                  epochs=args.epochs_m)