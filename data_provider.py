import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class DataProvider(Dataset):

    def __init__(self, inputs: np.ndarray, cond_inputs: np.ndarray = None):
        self.inputs = inputs
        if cond_inputs is not None:
            self.cond_inputs = cond_inputs
        else:
            self.cond_inputs = None

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        if self.cond_inputs is None:
            return self.inputs[idx, :]
        else:
            return [self.inputs[idx, :], self.cond_inputs[idx, :]]


def split_data_marginal(inputs, batch_size, num_workers=12):
    # Transform into data object
    data_train, data_val = train_test_split(inputs, test_size=0.2)
    data_val, data_test = train_test_split(inputs, test_size=0.5)

    data_train = DataProvider(data_train)
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            
    data_val = DataProvider(data_val)
    loader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    data_test = DataProvider(data_test)
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            
    return loader_train, loader_val, loader_test


def split_data_copula(x_inputs, y_inputs, cond_set, batch_size, num_workers):
    inputs_cond = torch.cat([x_inputs, y_inputs, cond_set], axis=1)

    data_train, data_val = train_test_split(inputs_cond, test_size=0.20)
    data_val, data_test = train_test_split(data_val, test_size=0.50)

    data_train = DataProvider(inputs=data_train[:, :2], cond_inputs=data_train[:, 2:])
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=0 if x_inputs.is_cuda else num_workers)


    data_val = DataProvider(inputs=data_val[:, :2], cond_inputs=data_val[:, 2:])
    loader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=0 if x_inputs.is_cuda else num_workers)


    data_test = DataProvider(inputs=data_test[:, :2], cond_inputs=data_test[:, 2:])
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=0 if x_inputs.is_cuda else num_workers)

    return loader_train, loader_val, loader_test