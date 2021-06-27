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
from statsmodels.distributions.empirical_distribution import ECDF
from utils import create_folders, random_search
from data_provider import split_data_marginal, Marginal_Distr
from options import TrainOptions
from eval.plots import visualize1d
eps = 1e-10


def ecdf_transform(data_train, data_test):
    norm_distr = scipy.stats.norm()
    ecdf = ECDF(data_train.reshape(-1,))
    uniform_samples = ecdf(data_test.reshape(-1,))
    uniform_samples[uniform_samples == 0] = eps
    uniform_samples[uniform_samples == 1] = 1 - eps
    gaussian_samples = norm_distr.ppf(uniform_samples)
    return -np.mean(norm_distr.logpdf(gaussian_samples))

def exp_marg_transform(inputs):

    # Training settings
    args = TrainOptions().parse()   # get training options
    args.exp_name = 'exp_marg_flow'

    # Create Folders
    create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.cuda.current_device()

    # Set Seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Transform into data object
    data_train, data_val, data_test, loader_train, loader_val, loader_test = split_data_marginal(inputs, batch_size=128, num_workers=0, return_datasets=True)

    # Run experiment 
    experiment, experiment_metrics, test_metrics = marginal_estimator(loader_train=loader_train, 
                                                                      loader_val=loader_val, 
                                                                      loader_test=loader_test, 
                                                                      exp_name=args.exp_name, 
                                                                      device=args.device, 
                                                                      amsgrad=args.amsgrad_m, 
                                                                      epochs=args.epochs, 
                                                                      num_workers=args.num_workers, 
                                                                      variable_num=0,
                                                                      n_layers=args.n_layers_m,
                                                                      lr=args.lr_m,
                                                                      weight_decay=args.weight_decay_m)

    # Plot results
    visualize1d(experiment.model, device=args.device, path=experiment.figures_path, true_samples=data_train, obs=1000, name='marg_flow')

    experiment_logs = os.path.join('results', args.exp_name, 'mf_0', 'stats')

    # Comparison to empirical CDF Transform:
    ecdf_nll = ecdf_transform(data_train, data_test)
    test_metrics['ecdf_nll'] = [ecdf_nll]

    print('Flow NLL: {}, ECDF NLL: {}'.format(test_metrics['test_loss'][0], ecdf_nll))
    save_statistics(experiment_logs, 'test_summary.csv', test_metrics, current_epoch=0, continue_from_mode=False)


if __name__ == '__main__':
    #exp_marg_transform()

    # Training settings
    args = TrainOptions().parse()   # get training options
    args.exp_name = 'exp_marg_flow'

    # Create Folders
    create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.cuda.current_device()

    # Set Seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Get inputs
    marginal_data = Marginal_Distr(args.marginal, mu=args.mu, var=args.var, alpha=args.alpha, low=args.low, high=args.high)
    inputs = marginal_data.sample(args.obs)
    
    exp_marg_transform(inputs)

    experiment_logs = os.path.join('results', args.exp_name, 'mf_0', 'stats')

    # Transform into data object
    data_train, data_val, data_test, loader_train, loader_val, loader_test = split_data_marginal(inputs, batch_size=128, num_workers=0, return_datasets=True)

    #random_search(loader_train, loader_val, loader_test, args.device, experiment_logs, iterations=200, epochs=50)
    random_search(marginal_estimator, 'random_search_marg', loader_train, loader_val, loader_test, args.device, experiment_logs, iterations=200, epochs=args.epochs)