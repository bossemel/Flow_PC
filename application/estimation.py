import os
import pandas as pd
from options import TrainOptions
from utils import HiddenPrints, create_folders
import torch
import json
import os 
from utils import create_folders, set_seeds
from options import TrainOptions
import time 
import pandas as pd
import networkx as nx
from pc import pc_estimator
from utils.pc_utils import resit
from eval.plots import plot_graph, visualize_joint
from cond_indep_test import copula_indep_test, marginal_transform
from utils import HiddenPrints


def pc_application(input_dataset: pd.DataFrame, indep_test, alpha, device, exp_name, 
           figures_path, kwargs_m=None, kwargs_c=None, add_name='') -> float:
    """
    Calculate the PC of a given graph.
    :param input_dataset: the dataset to be used for the PC calculation.
    :param target_graph: the target graph to be used for the SHD calculation.
    :param indep_test: the function to be used to calculate the independent test.
    :return: the shd between the estimated and the target graph.
    """
    # Plot the graph
    visualize_joint(input_dataset.iloc[:, 0:2].to_numpy(), figures_path, 'input_dataset')

    # Estimate the graph
    estimated_graph = pc_estimator(input_dataset, indep_test=indep_test, alpha=alpha, exp_name=exp_name,
                                    kwargs_m=kwargs_m, kwargs_c=kwargs_c, device=device)

    # Plot the estimated graph
    undirected_graph = estimated_graph.to_undirected()
    nx.draw(undirected_graph)
    plot_graph(undirected_graph, os.path.join(figures_path, add_name + 'est_graph.pdf'))
    nx.write_gpickle(undirected_graph, os.path.join(figures_path, add_name + 'est_graph.pickle'))


def reciprocity_exp(data, obs, test=False):
    data = data.filter(items=['log_concessions', 
                              'log_opp_concessions',
                              'log_offr_price',
                              'log_time_since_offer',
                              'log_hist',
                              'log_opp_hist']).dropna()

    if len(data) > obs:
        data_small = data.sample(obs)
    else:
        data_small = data
    del data
    print(data_small.columns)
    print(data_small.head())

    # Training settings
    args = TrainOptions().parse()
    args.exp_name = 'ebay_pc_recipr'
    args.flow_name = 'resit'
    args.alpha_indep = 0.05

    # Create Folders
    args = create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.cuda.current_device()

    # Set Seed
    set_seeds(seed=args.seed, use_cuda=use_cuda)

    # Run the PC algorithm
    start = time.time()
    pc_application(data_small, indep_test=resit, alpha=args.alpha_indep, 
            figures_path=args.figures_path, exp_name=args.exp_name,
           device=args.device, add_name='resit')
    end = time.time()
    print('Elapsed time: {}'.format(end - start))

    # Run the PC algorithm with the Flow-based independence test
    if test:
        args.epochs_m = 1
        args.epochs_c = 1
    # kwargs marginal
    kwargs_m = {'n_layers': args.n_layers_m,
              'lr': args.lr_m,
              'weight_decay': args.weight_decay_m,
              'amsgrad': args.amsgrad_m,
              'n_bins': args.n_bins_m,
              'tail_bound': args.tail_bound_m,
              'hidden_units': args.hidden_units_m,
              'tails': args.tails_m,
              'identity_init': args.identity_init_m,
              'epochs': args.epochs_m}

    # kwargs copula
    kwargs_c = {'n_layers': args.n_layers_c,
              'lr': args.lr_c,
              'weight_decay': args.weight_decay_c,
              'amsgrad': args.amsgrad_c,
              'n_bins': args.n_bins_c,
              'tail_bound': args.tail_bound_c,
              'hidden_units': args.hidden_units_m,
              'tails': args.tails_m,
              'identity_init': args.identity_init_m,
              'epochs': args.epochs_c,
              'n_blocks': args.n_blocks_c,
              'dropout': args.dropout_c,
              'use_batch_norm': args.batch_norm_c,
              'unconditional_transform': args.unconditional_transform_c}

    # Create new folders
    args.exp_name = 'ebay_pc_recipr'
    args.flow_name = 'cop_flow'
    args.alpha_indep = 0.05

    # Create Folders
    args = create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    start = time.time()

    transform = True
    if transform:
        # Transform marginals 
        print('Starting marginal transform')
        with HiddenPrints():
            columns = data_small.columns
            data_small = pd.DataFrame(marginal_transform(data_small.to_numpy(), args.exp_name, device=args.device, disable_tqdm=True, **kwargs_m).float().detach().cpu().numpy(),
                                        columns=columns)
        
        # Save transformed dataset
        data_small.to_csv(os.path.join(args.experiment_logs, 'transformed_data_small.csv'), index=False)

    # Load the transformed dataset
    data_small = pd.read_csv(os.path.join(args.experiment_logs, 'transformed_data_small.csv'))

    pc_application(data_small, indep_test=copula_indep_test, 
           alpha=args.alpha_indep, device=args.device, exp_name=args.exp_name,
           kwargs_m=kwargs_m, kwargs_c=kwargs_c, figures_path=args.figures_path, add_name='flow')
    end = time.time()
    print('Elapsed time: {}'.format(end - start))


def timing_exp(data, obs, test=False):
    data = data.filter(items=['log_concessions', 
                              'log_opp_concessions', 
                              'log_offr_price', 
                              'log_response_time',
                              'log_opp_response_time', 
                              'log_time_since_offer']).dropna()
    
    if len(data) > obs:
        data_small = data.sample(obs)
    else:
        data_small = data
    del data
    print(data_small.columns)
    print(data_small.head())

    # Training settings
    args = TrainOptions().parse()
    args.exp_name = 'ebay_pc_timing'
    args.flow_name = 'resit'
    args.alpha_indep = 0.05

    # Create Folders
    args = create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.cuda.current_device()

    # Set Seed
    set_seeds(seed=args.seed, use_cuda=use_cuda)

    # Run the PC algorithm
    start = time.time()
    pc_application(data_small, indep_test=resit, alpha=args.alpha_indep, 
            figures_path=args.figures_path, exp_name=args.exp_name,
           device=args.device, add_name='resit')
    end = time.time()
    print('Elapsed time: {}'.format(end - start))

    # Run the PC algorithm with the Flow-based independence test
    if test:
        args.epochs_m = 1
        args.epochs_c = 1
    # kwargs marginal
    kwargs_m = {'n_layers': args.n_layers_m,
              'lr': args.lr_m,
              'weight_decay': args.weight_decay_m,
              'amsgrad': args.amsgrad_m,
              'n_bins': args.n_bins_m,
              'tail_bound': args.tail_bound_m,
              'hidden_units': args.hidden_units_m,
              'tails': args.tails_m,
              'identity_init': args.identity_init_m,
              'epochs': args.epochs_m}

    # kwargs copula
    kwargs_c = {'n_layers': args.n_layers_c,
              'lr': args.lr_c,
              'weight_decay': args.weight_decay_c,
              'amsgrad': args.amsgrad_c,
              'n_bins': args.n_bins_c,
              'tail_bound': args.tail_bound_c,
              'hidden_units': args.hidden_units_m,
              'tails': args.tails_m,
              'identity_init': args.identity_init_m,
              'epochs': args.epochs_c,
              'n_blocks': args.n_blocks_c,
              'dropout': args.dropout_c,
              'use_batch_norm': args.batch_norm_c,
              'unconditional_transform': args.unconditional_transform_c}

    # Create new folders
    args.exp_name = 'ebay_pc_timing'
    args.flow_name = 'cop_flow'
    args.alpha_indep = 0.05

    # Create Folders
    args = create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    start = time.time()

    transform = True
    if transform:
        # Transform marginals 
        print('Starting marginal transform')
        with HiddenPrints():
            columns = data_small.columns
            data_small = pd.DataFrame(marginal_transform(data_small.to_numpy(), args.exp_name, device=args.device, disable_tqdm=True, **kwargs_m).float().detach().cpu().numpy(),
                                        columns=columns)
        
        # Save transformed dataset
        data_small.to_csv(os.path.join(args.experiment_logs, 'transformed_data_small.csv'), index=False)

    # Load the transformed dataset
    data_small = pd.read_csv(os.path.join(args.experiment_logs, 'transformed_data_small.csv'))

    pc_application(data_small, indep_test=copula_indep_test, 
           alpha=args.alpha_indep, device=args.device, exp_name=args.exp_name,
           kwargs_m=kwargs_m, kwargs_c=kwargs_c, figures_path=args.figures_path, add_name='flow')
    end = time.time()
    print('Elapsed time: {}'.format(end - start))


def reciprocity_t4_exp(data, obs, test=False):
    data = data[data['offer_counter'] >= 4]

    data = data.filter(items=['log_concessions', 
                              'log_opp_concessions',
                              'log_offr_price',
                              'log_time_since_offer',
                              'log_hist',
                              'log_opp_hist']).dropna()

    if len(data) > obs:
        data_small = data.sample(obs)
    else:
        data_small = data
    del data
    # Training settings
    args = TrainOptions().parse()
    args.exp_name = 'ebay_pc_recipr_t4'
    args.flow_name = 'resit'
    args.alpha_indep = 0.05

    # Create Folders
    args = create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.cuda.current_device()

    # Set Seed
    set_seeds(seed=args.seed, use_cuda=use_cuda)

    # Run the PC algorithm
    start = time.time()
    pc_application(data_small, indep_test=resit, alpha=args.alpha_indep, 
            figures_path=args.figures_path, exp_name=args.exp_name,
           device=args.device, add_name='resit')
    end = time.time()
    print('Elapsed time: {}'.format(end - start))

    # Run the PC algorithm with the Flow-based independence test
    if test:
        args.epochs_m = 1
        args.epochs_c = 1
    # kwargs marginal
    kwargs_m = {'n_layers': args.n_layers_m,
              'lr': args.lr_m,
              'weight_decay': args.weight_decay_m,
              'amsgrad': args.amsgrad_m,
              'n_bins': args.n_bins_m,
              'tail_bound': args.tail_bound_m,
              'hidden_units': args.hidden_units_m,
              'tails': args.tails_m,
              'identity_init': args.identity_init_m,
              'epochs': args.epochs_m}

    # kwargs copula
    kwargs_c = {'n_layers': args.n_layers_c,
              'lr': args.lr_c,
              'weight_decay': args.weight_decay_c,
              'amsgrad': args.amsgrad_c,
              'n_bins': args.n_bins_c,
              'tail_bound': args.tail_bound_c,
              'hidden_units': args.hidden_units_m,
              'tails': args.tails_m,
              'identity_init': args.identity_init_m,
              'epochs': args.epochs_c,
              'n_blocks': args.n_blocks_c,
              'dropout': args.dropout_c,
              'use_batch_norm': args.batch_norm_c,
              'unconditional_transform': args.unconditional_transform_c}

    # Create new folders
    args.exp_name = 'ebay_pc_recipr_t4'
    args.flow_name = 'cop_flow'
    args.alpha_indep = 0.05

    # Create Folders
    args = create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    start = time.time()

    transform = True
    if transform:
        # Transform marginals 
        print('Starting marginal transform')
        with HiddenPrints():
            columns = data_small.columns
            data_small = pd.DataFrame(marginal_transform(data_small.to_numpy(), args.exp_name, device=args.device, disable_tqdm=True, **kwargs_m).float().detach().cpu().numpy(),
                                        columns=columns)
        
        # Save transformed dataset
        data_small.to_csv(os.path.join(args.experiment_logs, 'transformed_data_small.csv'), index=False)

    # Load the transformed dataset
    data_small = pd.read_csv(os.path.join(args.experiment_logs, 'transformed_data_small.csv'))

    pc_application(data_small, indep_test=copula_indep_test, 
           alpha=args.alpha_indep, device=args.device, exp_name=args.exp_name,
           kwargs_m=kwargs_m, kwargs_c=kwargs_c, figures_path=args.figures_path, add_name='flow')
    end = time.time()
    print('Elapsed time: {}'.format(end - start))


if __name__ == '__main__':
 
    # Load dataset
    file_path_cons = os.path.join('datasets', 'ebay_data', 'consessions_subset.csv')
    data = pd.read_csv(file_path_cons, index_col=0)

    data['log_concessions_by_offr_price'] = data['log_concessions'] - data['log_offr_price']
    data['log_opp_concessions_by_offr_price'] = data['log_opp_concessions'] - data['log_offr_price']

    data['log_hist'] = data['log_hist'].astype('float')
    data['log_opp_hist'] = data['log_opp_hist'].astype('float')
    data['log_concessions_by_offr_price'] = data['log_concessions_by_offr_price'].astype('float')
    data['log_opp_concessions_by_offr_price'] = data['log_opp_concessions_by_offr_price'].astype('float')
    data['offer_counter'] = data['offer_counter'].astype('float')

    obs = 10000
    test = False

    # reciprocity_exp(data, obs, test=test)
    # reciprocity_t4_exp(data, obs, test=test)
    timing_exp(data, obs, test=test)

    # # Create new folders
    # args = TrainOptions().parse()
    # args.exp_name = 'ebay_pc_timing'
    # args.flow_name = 'cop_flow'
    # args.alpha_indep = 0.05
    # add_name = 'flow'

    # # Create Folders
    # args = create_folders(args)

    # # Read pickled graph
    # undirected_Graph = nx.read_gpickle(os.path.join(os.path.join(args.figures_path, add_name + 'est_graph.pickle')))

    # print(nx.info(undirected_Graph))

    # # Show the nodes
    # print(undirected_Graph.nodes())
    # print(undirected_Graph.edges())