import numpy as np
import torch
from torch.utils.data import Dataset


class DataProvider(Dataset): # @Todo: debug this. possibly use pytorch data loader class...

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
            return self.inputs[idx, :], self.cond_inputs[idx, :]
