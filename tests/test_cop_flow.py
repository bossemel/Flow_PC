# @Todo: write test function that trains and evaluates conditional copula flow for conditional copula inputs
from cond_indep_test import copula_estimator
from options import TrainOptions
from utils import create_folders
import torch
import numpy as np
import json
import random
import os 

def test_copula_estimator():

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
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Get inputs
    obs = 50
    args.epochs = 1
    xx = torch.from_numpy(np.random.uniform(size=(obs, 1)))
    yy = torch.from_numpy(np.random.uniform(size=(obs, 1)))
    zz = torch.from_numpy(np.random.uniform(size=(obs, 5)))

    copula_estimator(xx, yy, zz, 'copula_flow_test', args.device)

if __name__ == '__main__':
    test_copula_estimator()