import torch.optim as optim
import torch
import os
from pathlib import Path
from nflows import flows
import torch.nn as nn
from sklearn.model_selection import train_test_split
from data_provider import DataProvider
from torch.utils.data import DataLoader


def set_optimizer_scheduler(model, lr, weight_decay, amsgrad, epochs):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler

def nll_error(log_density):
    return -torch.mean(log_density)

def create_folders(args):
    args.exp_path = os.path.join('results', args.exp_name)
    args.figures_path = os.path.join(args.exp_path, 'figures')
    args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    args.experiment_saved_models = os.path.join('saved_models', args.exp_name)
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_saved_models).mkdir(parents=True, exist_ok=True)


class Flow_decorator(flows.Flow):
    def __init__(self, transform, distribution):
        super().__init__(transform=transform, distribution=distribution)
        self.layer_dict = nn.ModuleDict()

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass


def split_data_marginal(inputs, batch_size, num_workers=12):
    # Transform into data object
    data_train, data_val = train_test_split(inputs, test_size=0.2)
    data_val, data_test = train_test_split(inputs, test_size=0.5)

    data_train = DataProvider(data_train)
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                                        # @Todo: take these arguments from somewhere else
    data_val = DataProvider(data_val)
    loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    data_test = DataProvider(data_test)
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                                        # @Todo: take these arguments from somewhere else
    # @Todo: create test set loader
    return loader_train, loader_val, loader_test


def split_data_copula(inputs_cond, batch_size, num_workers):
    data_train, data_val = train_test_split(inputs_cond, test_size=0.20)
    data_val, data_test = train_test_split(data_val, test_size=0.50)

    data_train = DataProvider(inputs=data_train[:, :2], cond_inputs=data_train[:, 2:])
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                            # @Todo: take these arguments from somewhere else

    data_val = DataProvider(inputs=data_val[:, :2], cond_inputs=data_val[:, 2:])
    loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                            # @Todo: take these arguments from somewhere else

    data_test = DataProvider(inputs=data_test[:, :2], cond_inputs=data_test[:, 2:])
    loader_test = DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                            # @Todo: take these arguments from somewhere else
    return loader_train, loader_val, loader_test

