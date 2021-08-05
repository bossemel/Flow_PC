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

eps = 1e-7


def pc_exp(input_dataset: pd.DataFrame, target_graph: nx.Graph, indep_test, alpha, device, exp_name, 
           figures_path, kwargs_m=None, kwargs_c=None, add_name='') -> float:
    """
    Calculate the PC of a given graph.
    :param input_dataset: the dataset to be used for the PC calculation.
    :param target_graph: the target graph to be used for the SHD calculation.
    :param indep_test: the function to be used to calculate the independent test.
    :return: the shd between the estimated and the target graph.
    """
    # Plot the graph
    plot_graph(target_graph, os.path.join(figures_path, add_name + 'sachs_graph.pdf'))
    visualize_joint(input_dataset[['PKC', 'PKA']].to_numpy(), figures_path, 'input_dataset')

    # Estimate the graph
    estimated_graph = pc_estimator(input_dataset, indep_test=indep_test, alpha=alpha, exp_name=exp_name,
                                    kwargs_m=kwargs_m, kwargs_c=kwargs_c, device=device)

    # Plot the estimated graph
    undirected_graph = estimated_graph.to_undirected()
    nx.draw(undirected_graph)
    plot_graph(undirected_graph, os.path.join(figures_path, add_name + 'est_graph.pdf'))
    
    # Calculate the structural hamming distance
    shd = shd_calculator(target_graph, estimated_graph)

    return shd


if __name__ == '__main__':
    # Training settings
    args = TrainOptions().parse()
    args.exp_name = 'exp_pc'
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


    s_data, s_graph = load_dataset('sachs')

    # Create smaller dataset
    s_data_small = s_data[['PKC', 'PKA', 'praf', 'pmek', 'p44/42']]
    s_graph_small = s_graph.subgraph(s_data_small.columns)

    # Run the PC algorithm
    start = time.time()
    pc_exp(s_data_small, s_graph_small, indep_test=resit, alpha=args.alpha_indep, 
            figures_path=args.figures_path, exp_name=args.exp_name,
           device=args.device, add_name='resit')
    end = time.time()
    print('Elapsed time: {}'.format(end - start))

    # Run the PC algorithm with the Flow-based independence test
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

    # # Create new folders
    # args.exp_name = 'exp_pc'
    # args.flow_name = 'cop_flow'
    # args.alpha_indep = 0.05

    # # Create Folders
    # args = create_folders(args)
    # with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)

    # start = time.time()
    # pc_exp(s_data_small, s_graph_small, indep_test=copula_indep_test, 
    #        alpha=args.alpha_indep, device=args.device, exp_name=args.exp_name,
    #        kwargs_m=kwargs_m, kwargs_c=kwargs_c, figures_path=args.figures_path, add_name='flow')
    # end = time.time()
    # print('Elapsed time: {}'.format(end - start))
