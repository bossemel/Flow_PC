from cond_indep_test import marginal_transform_1d
from options import TrainOptions
from utils import create_folders, set_seeds
import torch
import numpy as np
import json
import random
import os 
from eval.plots import visualize1d


def test_marginal_estimator():

    # Training settings
    args = TrainOptions().parse()   # get training options

    # Create Folders
    create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.cuda.current_device()

    # Set Seed
    set_seeds(args.seed)

    # Get inputs
    obs = 50
    args.epochs = 1
    xx = np.random.uniform(size=(obs, 1))

    kwargs = {'n_layers': args.n_layers_m,
              'lr': args.lr_m,
              'weight_decay': args.weight_decay_m,
              'amsgrad': args.amsgrad_m,
              'n_bins': args.n_bins_m,
              'tail_bound': args.tail_bound_m,
              'identity_init': args.identity_init_m,
              'hidden_units': args.hidden_units_m,
              'tails': args.tails_m}

    outputs = marginal_transform_1d(xx, 'test_marg_flow', args.device, **kwargs)


if __name__ == '__main__':
    test_marginal_estimator()