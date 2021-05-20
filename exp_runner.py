import torch
import torch.nn as nn
import numpy as np
import tqdm
from collections import OrderedDict
import logging
import time

logger = logging.getLogger(__name__)


class Experiment:

    def __init__(self, model: nn.Module, optimizer, scheduler, train_dataset,
                 valid_dataset=None, data_monitors: dict = None, clip_grad_norm=False, clip: float = 5):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.data_monitors = OrderedDict()
        if data_monitors is not None:
            self.data_monitors.update(data_monitors)
        self.tqdm_progress = tqdm.tqdm
        self.clip_grad_norm = clip_grad_norm
        self.clip = clip

    def do_training_epoch(self):
        with self.tqdm_progress(total=self.train_dataset.num_batches) as train_progress_bar:
            train_progress_bar.set_description("Epoch")
            for inputs_batch in self.train_dataset:
                log_density = self.model.log_prob(inputs_batch) #@Todo: put model into eval mode somehow
                loss = -torch.mean(log_density)
                #loss = self.model.loss(inputs_batch)
                loss.backward()

                # Perform gradient clipping
                if self.clip_grad_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()
                self.scheduler.step()
                train_progress_bar.update(1)

    def eval_monitors(self, dataset, label):
        data_mon_vals = OrderedDict([(key + label, 0.) for key
                                     in self.data_monitors.keys()])
        for inputs_batch in dataset:
            log_density = self.model.log_prob(inputs_batch) #@Todo: put model into eval mode somehow
            loss = -torch.mean(log_density)
            # loss = self.model.loss(inputs_batch, evaluation=True)
            for key, data_monitor in self.data_monitors.items():
                data_mon_vals[key + label] += data_monitor(
                    loss)
        for key, data_monitor in self.data_monitors.items():
            data_mon_vals[key + label] /= dataset.num_batches
        return data_mon_vals

    def get_epoch_stats(self):
        epoch_stats = OrderedDict()
        epoch_stats.update(self.eval_monitors(self.train_dataset, '(train)'))
        if self.valid_dataset is not None:
            epoch_stats.update(self.eval_monitors(
                self.valid_dataset, '(valid)'))
        return epoch_stats

    def log_stats(self, epoch, epoch_time, stats):
        logger.info('Epoch {0}: {1:.1f}s to complete\n    {2}'.format(
            epoch, epoch_time,
            ', '.join(['{}={:.2e}'.format(k, v) for (k, v) in stats.items()])
        ))

    def train(self, num_epochs, stats_interval=5):
        start_train_time = time.time()
        run_stats = [list(self.get_epoch_stats().values())]
        with self.tqdm_progress(total=num_epochs) as progress_bar:
            progress_bar.set_description("Experiment")
            for epoch in range(1, num_epochs + 1):
                start_time = time.time()
                self.do_training_epoch()
                epoch_time = time.time() - start_time
                if epoch % stats_interval == 0:
                    stats = self.get_epoch_stats()
                    self.log_stats(epoch, epoch_time, stats)
                    run_stats.append(list(stats.values()))
                progress_bar.update(1)
        finish_train_time = time.time()
        total_train_time = finish_train_time - start_train_time
        return np.array(run_stats), {k: i for i, k in enumerate(stats.keys())}, total_train_time
