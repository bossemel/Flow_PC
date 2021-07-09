# @Todo: write test function that trains and evaluates conditional copula flow for conditional copula inputs
from cond_indep_test import copula_estimator, mi_estimator, hypothesis_test
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
from utils import create_folders, random_search, set_seeds, kde_estimator
from data_provider import split_data_copula, Copula_Distr, mutivariate_copula
from options import TrainOptions
from eval.plots import visualize_joint
from eval.metrics import jsd_copula, jsd_copula_context
import statsmodels.api
import tqdm
eps = 1e-10


# @Todo: implement comparison (copula? kde?) e.g. statsmodels.nonparametric.kernel_density.KDEMultivariateConditional

def exp_cop_transform(inputs: torch.Tensor, copula_distr, cond_inputs: torch.Tensor =None, cond_copula_distr=None):
    
    cond_set_dim = cond_inputs.shape[-1] if cond_inputs is not None else None

    # Transform into data object
    visualize_joint(inputs, args.figures_path, name='input_dataset')
    norm_distr = scipy.stats.norm()
    visualize_joint(norm_distr.cdf(inputs), args.figures_path, name='input_uni_dataset')
    if cond_inputs is not None:
        visualize_joint(cond_inputs, args.figures_path, name='cond_input_dataset')
        visualize_joint(norm_distr.cdf(cond_inputs), args.figures_path, name='cond_input_uni_dataset')

    data_train, __, __, loader_train, loader_val, loader_test = split_data_copula(inputs[:, 0:1], 
                                                                                  inputs[:, 1:2], 
                                                                                  cond_inputs, 
                                                                                  batch_size=128, 
                                                                                  num_workers=0, 
                                                                                  return_datasets=True)

    # Run experiment 
    experiment, __, test_metrics = copula_estimator(loader_train=loader_train, 
                                                    loader_val=loader_val, 
                                                    loader_test=loader_test, 
                                                    cond_set_dim=cond_set_dim,
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
                                                    weight_decay=args.weight_decay_c,
                                                    n_blocks=args.n_blocks_c,
                                                    dropout=args.dropout_c,
                                                    use_batch_norm=args.batch_norm_c,
                                                    tails=args.tails_c,
                                                    n_bins=args.n_bins_c,
                                                    unconditional_transform=args.unconditional_transform_c)
    # @Todo: recheck: is the flow in eval mode here?
    if cond_inputs is not None:
        normal_distr = torch.distributions.normal.Normal(0, 1)
        cond_inputs = normal_distr.cdf(cond_inputs).float().to(args.device)

    # Plot results
    with torch.no_grad(): 
        samples = experiment.model.sample_copula(inputs.shape[0], context=cond_inputs).cpu().numpy()
    visualize_joint(samples, experiment.figures_path, name=args.exp_name)

    # Calculate JSD # @Todo: figure out conditional inputs # @Todo: watch out: can't simulate conditionally from data
    with torch.no_grad():
        if cond_set_dim is None:
            jsd = jsd_copula(experiment.model, copula_distr, args.device, num_samples=100000)
        else:
            jsd = jsd_copula_context(experiment.model, copula_distr, args.device, context=cond_inputs, num_samples=cond_inputs.shape[0], cond_copula_distr=cond_copula_distr.pdf)
            
    test_metrics['cop_flow_jsd'] = [jsd.item()]
    experiment_logs = os.path.join('results', args.exp_name, 'cf', 'stats')

    # # Comparison to empirical CDF Transform:
    print(copula_distr.theta)
    test_metrics['kde_jsd'] = kde_estimator(data_train, copula_distr, 100000, args.device) # @Todo: write conditional kde estimator

    print('Flow JSD: {}, KDE JSD: {}'.format(test_metrics['cop_flow_jsd'][0], test_metrics['kde_jsd'][0]))
    num_runs = 1

    print('Estimating mutual information..')
    with torch.no_grad():
        experiment.model.eval()
        mi_runs = []
        ii = 0
        with tqdm.tqdm(total=num_runs) as pbar_test:  # ini a progress bar
            while ii < num_runs:
                mi_estimate = mi_estimator(experiment.model, device=args.device, cond_set_dim=cond_set_dim, obs_n=10000, obs_m=1000)
                if not np.isnan(mi_estimate):
                    mi_runs.append(mi_estimate)
                    ii += 1
                pbar_test.update(1)

    test_metrics['mi_runs_mean'] = [np.mean(mi_runs)]
    test_metrics['mi_runs_std'] = [np.std(mi_runs)]
    print('Running hypothesis test..')
    result = hypothesis_test(np.array(mi_runs), threshold=0.05)
    test_metrics['independent'] = [result]
    print(test_metrics)
    save_statistics(experiment_logs, 'test_summary.csv', test_metrics, current_epoch=0, continue_from_mode=False)


def exp_2D_cop(args):
    experiments = [('indep_2D', 'independent', None),
                   ('clayton_con_2D', 'clayton', 2), 
                   ('clayton_uncon_2D', 'clayton', 0+eps), 
                   ('frank_con_2D', 'frank', 5), 
                   ('frank_uncon_2D', 'frank', 0+eps),
                   ('gumbel_con_2D', 'gumbel', 5), 
                   ('gumbel_uncon_2D', 'gumbel', 1+eps)]

    for experiment in experiments:
        args.exp_name = 'exp_cop_flow' + experiment[0]
        print('Starting {}'.format(args.exp_name))
        args.copula = experiment[1]
        args.theta = experiment[2]

        # Create Folders
        args = create_folders(args)
        with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # Get inputs
        copula_distr = Copula_Distr(args.copula, theta=args.theta, transform=False)
        inputs = torch.from_numpy(copula_distr.sample(args.obs)) # @Todo: create conditional inputs
        normal_distr = torch.distributions.normal.Normal(0, 1)
        inputs = normal_distr.icdf(inputs).float()

        exp_cop_transform(inputs, copula_distr)


def exp_4D_cop(args):
    experiments = [('clayton_con_4D', 'clayton', 2), 
                   ('clayton_uncon_4D', 'clayton', 0+eps), 
                   ('frank_con_4D', 'frank', 5), 
                   ('frank_uncon_4D', 'frank', 0+eps),
                   ('gumbel_con_4D', 'gumbel', 5), 
                   ('gumbel_uncon_4D', 'gumbel', 1+eps)]

    for experiment in experiments:
        args.exp_name = 'exp_cop_flow' + experiment[0]
        args.copula = experiment[1]
        args.theta = experiment[2]

        # Create Folders
        args = create_folders(args)
        with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        # Get inputs
        inputs, copula_distr = mutivariate_copula(mix=False, copula=experiment[1], marginal=None, theta=experiment[2], num_samples=args.obs, disable_marginal=True)
        normal_distr = torch.distributions.normal.Normal(0, 1)
        inputs = normal_distr.icdf(inputs).float()
        cond_copula_distr = Copula_Distr(args.copula, theta=args.theta, transform=True)
        # @Todo: debug this
        exp_cop_transform(inputs[:, 0:2], copula_distr, inputs[:, 2:4], cond_copula_distr=cond_copula_distr)
        exit()

if __name__ == '__main__':
    # Training settings
    args = TrainOptions().parse()   # get training options
    args.exp_name = 'exp_cop_flow'
    args.flow_name = 'cf'

    # Create Folders
    args = create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.cuda.current_device()

    # Set Seed
    set_seeds(seed=args.seed, use_cuda=use_cuda)

    exp_2D_cop(args)
    #exp_4D_cop(args)
    exit()
    #Get inputs
    copula_distr = Copula_Distr(args.copula, theta=args.theta, transform=True)
    inputs = torch.from_numpy(copula_distr.sample(args.obs)) # @Todo: create conditional inputs
    normal_distr = torch.distributions.normal.Normal(0, 1)
    inputs = normal_distr.icdf(inputs).float()
    
    #exp_cop_transform(inputs, copula_distr)
    #exit()

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