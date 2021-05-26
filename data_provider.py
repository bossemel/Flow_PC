import numpy as np
import torch
from torch.utils.data import Dataset


class DataProvider(Dataset): # @Todo: debug this. possibly use pytorch data loader class...

    def __init__(self, inputs: np.ndarray, cond_inputs: np.ndarray = None):
        self.inputs = torch.from_numpy(inputs).float()
        if cond_inputs is not None:
            self.cond_inputs = torch.from_numpy(cond_inputs).float()
        else:
            self.cond_inputs = None
        # if batch_size < 1:
        #     raise ValueError('batch_size must be >= 1')
        # self._batch_size = batch_size
        # if max_num_batches == 0 or max_num_batches < -1:
        #     raise ValueError('max_num_batches must be -1 or > 0')
        # self._max_num_batches = max_num_batches
        # self._update_num_batches()
        # self.shuffle_order = shuffle_order
        # self._current_order = np.arange(inputs.shape[0])
        # if random_seed is None:
        #     self.rng = np.random.RandomState(4)
        # else:
        #     self.rng = np.random.RandomState(random_seed)
        # self.new_epoch()

    # @property
    # def batch_size(self):
    #     return self._batch_size

    # @batch_size.setter
    # def batch_size(self, value):
    #     if value < 1:
    #         raise ValueError('batch_size must be >= 1')
    #     self._batch_size = value
    #     self._update_num_batches()

    # @property
    # def max_num_batches(self):
    #     return self._max_num_batches

    # @max_num_batches.setter
    # def max_num_batches(self, value):
    #     if value == 0 or value < -1:
    #         raise ValueError('max_num_batches must be -1 or > 0')
    #     self._max_num_batches = value
    #     self._update_num_batches()

    # def _update_num_batches(self):
    #     possible_num_batches = self.inputs.shape[0] // self.batch_size
    #     if self.max_num_batches == -1:
    #         self.num_batches = possible_num_batches
    #     else:
    #         self.num_batches = min(self.max_num_batches, possible_num_batches)

    # def __iter__(self):
    #     return self

    # def new_epoch(self):
    #     self._curr_batch = 0
    #     if self.shuffle_order:
    #         self.shuffle()

    # def __next__(self):
    #     return self.next()

    # def reset(self):
    #     """Resets the provider to the initial state."""
    #     inv_perm = np.argsort(self._current_order)
    #     self._current_order = self._current_order[inv_perm]
    #     self.inputs = self.inputs[inv_perm]
    #     if self.cond_inputs is not None:
    #         self.cond_inputs = self.cond_inputs[inv_perm]
    #     self.new_epoch()

    # def shuffle(self):
    #     """Randomly shuffles order of data."""
    #     perm = self.rng.permutation(self.inputs.shape[0])
    #     self._current_order = self._current_order[perm]
    #     self.inputs = self.inputs[perm]
    #     if self.cond_inputs is not None:
    #         self.cond_inputs = self.cond_inputs[perm]

    # def next(self):
    #     """Returns next data batch or raises `StopIteration` if at end."""
    #     if self._curr_batch + 1 > self.num_batches:
    #         # no more batches in current iteration through data set so start
    #         # new epoch ready for another pass and indicate iteration is at end
    #         self.new_epoch()
    #         raise StopIteration()
    #     # create an index slice corresponding to current batch number
    #     batch_slice = slice(self._curr_batch * self.batch_size,
    #                         (self._curr_batch + 1) * self.batch_size)
    #     inputs_batch = self.inputs[batch_slice]
    #     if self.cond_inputs is not None:
    #         cond_inputs_batch = self.cond_inputs[batch_slice]
    #     else:
    #         cond_inputs_batch = None
    #     self._curr_batch += 1
    #     return inputs_batch, cond_inputs_batch

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int):
        if self.cond_inputs is None:
            return self.inputs[idx, :]
        else:
            return self.inputs[idx, :], self.cond_inputs[idx, :]
