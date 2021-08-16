import os
import pandas as pd
import numpy as np 
from eval.plots import histogram
from pathlib import Path
from options import TrainOptions
from utils import create_folders
import torch
import json
import os 
from utils import create_folders, set_seeds
from options import TrainOptions
import time 
from cdt.data import load_dataset
from pc import pc_estimator, shd_calculator
import pandas as pd
import networkx as nx
from utils.pc_utils import pcalg, resit
from eval.plots import plot_graph, visualize_joint
from cond_indep_test import copula_indep_test


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



if __name__ == '__main__':
 
    # Load dataset
    file_path_cons = os.path.join('datasets', 'ebay_data', 'consessions_subset.csv')
    data = pd.read_csv(file_path_cons)
    
    data['log_concessions_by_offr_price'] = data['log_concessions'] - data['log_offr_price']
    data['log_opp_concessions_by_offr_price'] = data['log_opp_concessions'] - data['log_offr_price']

    data = data.filter(items=['log_concessions', 
                              'log_opp_concessions',
                              'offer_counter',
                              'log_slr_hist', 
                              'log_byr_hist']).dropna()
    data_small = data.head(10)
    del data
    print(data_small.columns)
    print(data_small.head())

    # Training settings
    args = TrainOptions().parse()
    args.exp_name = 'ebay_pc'
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
    args.exp_name = 'exp_pc'
    args.flow_name = 'cop_flow'
    args.alpha_indep = 0.05

    # Create Folders
    args = create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    start = time.time()
    pc_application(data_small, indep_test=copula_indep_test, 
           alpha=args.alpha_indep, device=args.device, exp_name=args.exp_name,
           kwargs_m=kwargs_m, kwargs_c=kwargs_c, figures_path=args.figures_path, add_name='flow')
    end = time.time()
    print('Elapsed time: {}'.format(end - start))
