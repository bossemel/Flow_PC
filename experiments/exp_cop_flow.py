from flows import Basic_Flow
from cond_indep_test import copula_estimator, mi_estimator, independence_test
from options import TrainOptions
from utils import create_folders
import torch
import numpy as np
import json
import os 
from utils.load_and_save import save_statistics
import scipy.stats
from utils import create_folders, random_search, set_seeds, kde_estimator
from data_provider import split_data_copula, Copula_Distr, mutivariate_copula
from options import TrainOptions
from eval.plots import visualize_joint
from eval.metrics import jsd_copula
import time 
eps = 1e-7


def visualize_inputs(inputs: np.ndarray, cond_inputs: np.ndarray) -> None:
    visualize_joint(inputs, args.figures_path, name='input_dataset')
    norm_distr = scipy.stats.norm()
    visualize_joint(norm_distr.cdf(inputs), args.figures_path, name='input_uni_dataset')
    if cond_inputs is not None:
        visualize_joint(cond_inputs, args.figures_path, name='cond_input_dataset')
        visualize_joint(norm_distr.cdf(cond_inputs), args.figures_path, name='cond_input_uni_dataset')


def cop_eval(model: Basic_Flow, inputs: torch.Tensor, cond_inputs: torch.Tensor, copula_distr: Copula_Distr, cond_copula_distr: Copula_Distr, 
             cond_set_dim: int, data_train: torch.Tensor, figures_path: str, experiment_logs: str) -> dict:
    eval_metrics = {}

    # Plot copula samples
    samples = model.sample_copula(inputs.shape[0], context=cond_inputs).cpu().numpy()
    visualize_joint(samples, figures_path, name=args.exp_name)

    # Calculate JSD
    if cond_set_dim is None:
        jsd = jsd_copula(model, copula_distr, args.device, num_samples=100000)
        eval_metrics['cop_flow_jsd'] = [jsd.item()]

    # # Comparison to empirical CDF Transform:
    if cond_set_dim is None:
        eval_metrics['kde_jsd'] = kde_estimator(data_train, copula_distr, 100000, args.device)
        print('Flow JSD: {}, KDE JSD: {}'.format(eval_metrics['cop_flow_jsd'][0], eval_metrics['kde_jsd'][0]))

    print('Estimating mutual information..')
    mi = mi_estimator(model, device=args.device, cond_set=cond_inputs)
    eval_metrics['mi'] = [mi.item()]
    print('MI: {}'.format(mi))
    result = independence_test(mi, threshold=0.05)

    eval_metrics['independent'] = [result]
    return eval_metrics


def exp_cop(inputs: torch.Tensor, copula_distr, cond_inputs: torch.Tensor =None, cond_copula_distr: Copula_Distr =None) -> None:
    # Set conditioning set dimnesion
    cond_set_dim = cond_inputs.shape[-1] if cond_inputs is not None else None

    # Visualize inputs
    visualize_inputs(inputs.cpu(), cond_inputs.cpu() if cond_inputs is not None else None)

    # Create data loaders
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

    # Evaluate
    experiment_logs = os.path.join('results', args.exp_name, 'cf', 'stats')

    with torch.no_grad(): 
        experiment.model.eval()
        eval_metrics = cop_eval(experiment.model, 
                                inputs, 
                                cond_inputs, 
                                copula_distr, 
                                cond_copula_distr, 
                                cond_set_dim,
                                data_train, 
                                experiment.figures_path,
                                experiment_logs)
        test_metrics.update(eval_metrics)

        # Saving results
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

        exp_cop(inputs, copula_distr)


def exp_4D_cop(args):
    args.n_layers = 1
    args.hidden_units = 16
    args.n_blocks = 3
    args.n_bins = 30
    args.dropout = 0.15
    args.lr = 0.01
    args.weight_decay = 1e-08
    args.tail_bound = 32
    args.batch_norm = False
    args.amsgrad = False
    args.clip_grad = False
    args.identity_init = True
    args.unconditional_transform = False

    experiments = [('indep_4D', 'independent', None), 
                   ('clayton_con_4D', 'clayton', 2), 
                   ('clayton_uncon_4D', 'clayton', 0+eps), 
                   ('frank_con_4D', 'frank', 5), 
                   ('frank_uncon_4D', 'frank', 0+eps),
                   ('gumbel_con_4D', 'gumbel', 5), 
                   ('gumbel_uncon_4D', 'gumbel', 1+eps)]

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
        inputs, copula_distr = mutivariate_copula(mix=False, copula=experiment[1], marginal=None, theta=experiment[2], num_samples=args.obs, disable_marginal=True)
        normal_distr = torch.distributions.normal.Normal(0, 1)
        inputs = normal_distr.icdf(inputs).float()
        cond_copula_distr = Copula_Distr(args.copula, theta=args.theta, transform=True)

        exp_cop(inputs[:, 0:2].to(args.device), copula_distr, inputs[:, 2:4].to(args.device), cond_copula_distr=cond_copula_distr)


def random_search_2D():
    #Get inputs
    copula_distr = Copula_Distr('independent', theta=None, transform=True)
    inputs = torch.from_numpy(copula_distr.sample(args.obs))
    normal_distr = torch.distributions.normal.Normal(0, 1)
    inputs = normal_distr.icdf(inputs).float()
    
    # Transform into data object
    __, __, __, loader_train, loader_val, loader_test = split_data_copula(inputs[:, 0:1], 
                                                                                               inputs[:, 1:2], 
                                                                                               None, 
                                                                                               batch_size=128, 
                                                                                               num_workers=0, 
                                                                                               return_datasets=True)

    # Run random search
    random_search(copula_estimator, 'random_search_cop', loader_train, loader_val,
     loader_test, args.device, args.experiment_logs, iterations=200, epochs=args.epochs_c, flow_type='cop_flow')


def random_search_4D():
    #Get inputs
    inputs, copula_distr = mutivariate_copula(mix=False, copula='independent', marginal=None, theta=None, num_samples=args.obs, disable_marginal=True)
    inputs = torch.from_numpy(copula_distr.sample(args.obs))
    normal_distr = torch.distributions.normal.Normal(0, 1)
    inputs = normal_distr.icdf(inputs).float()
    
    # Transform into data object
    __, __, __, loader_train, loader_val, loader_test = split_data_copula(inputs[:, 0:1], 
                                                                          inputs[:, 1:2], 
                                                                          inputs[:, 2:4],
                                                                          batch_size=128, 
                                                                          num_workers=0, 
                                                                          return_datasets=True)

    #random_search(loader_train, loader_val, loader_test, args.device, experiment_logs, iterations=200, epochs=50)
    random_search(copula_estimator, 'random_search_cop', loader_train, loader_val,
                  loader_test, args.device, args.experiment_logs, iterations=200, 
                  epochs=args.epochs_c, flow_type='cop_flow', cond_set_dim=2)


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

    #exp_2D_cop(args)
    #exp_4D_cop(args)

    #Get inputs
    copula_distr = Copula_Distr(args.copula, theta=args.theta, transform=True)
    inputs = torch.from_numpy(copula_distr.sample(args.obs)) # @Todo: create conditional inputs
    normal_distr = torch.distributions.normal.Normal(0, 1)
    inputs = normal_distr.icdf(inputs).float()
    start_time = time.time()
    exp_cop_transform(inputs, copula_distr)
    print("--- %s seconds ---" % (time.time() - start_time))
    #exit()

    #random_search_2D()
    #random_search_4D()
