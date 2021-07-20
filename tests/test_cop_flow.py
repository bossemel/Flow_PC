from cond_indep_test import copula_estimator
from options import TrainOptions
from utils import create_folders, set_seeds
import torch
import numpy as np
import json
import random
import os 
from data_provider import split_data_copula

def test_copula_estimator():

    # Training settings
    args = TrainOptions().parse()   # get training options
    args.exp_name = 'testing'
    args.flow_name = 'cf'

    # Create Folders
    create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    args.device = torch.cuda.current_device()

    # Set Seed
    set_seeds(args.seed)

    # Get inputs
    obs = 50
    args.epochs = 1
    xx = torch.from_numpy(np.random.uniform(size=(obs, 1)))
    yy = torch.from_numpy(np.random.uniform(size=(obs, 1)))
    zz = torch.from_numpy(np.random.uniform(size=(obs, 5)))

    kwargs = {'n_layers': args.n_layers_c,
              'lr': args.lr_c,
              'weight_decay': args.weight_decay_c,
              'amsgrad': args.amsgrad_c,
              'hidden_units': args.hidden_units_c,
              'tail_bound': args.tail_bound_c,
              'n_blocks': args.n_blocks_c,
              'dropout': args.dropout_c,
              'use_batch_norm': args.batch_norm_c,
              'tails': args.tails_c, 
              'n_bins': args.n_bins_c,
              'unconditional_transform': args.unconditional_transform_c}

    # Transform into data object
    __, __, __, loader_train, loader_val, loader_test = split_data_copula(xx, 
                                                                          yy, 
                                                                          zz, 
                                                                          batch_size=128, 
                                                                          num_workers=0, 
                                                                          return_datasets=True)

    copula_estimator(loader_train, 
                     loader_val, 
                     loader_test, 
                     cond_set_dim=zz.shape[-1], 
                     epochs=1, 
                     exp_name=args.exp_name, 
                     device=args.device, 
                     **kwargs)


if __name__ == '__main__':
    test_copula_estimator()