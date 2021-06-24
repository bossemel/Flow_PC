import torch
import torch.nn as nn
import tqdm
import os
import numpy as np
import time
from pathlib import Path

import matplotlib
matplotlib.rcParams.update({'font.size': 8})

from utils.load_and_save import save_statistics, load_statistics
from utils.plotting import plot_result_graphs


class ExperimentBuilder(nn.Module):
    def __init__(self, network_model, optimizer,
                 scheduler, error, exp_name, flow_name, epochs, train_data, val_data,
                 test_data, device, continue_from_epoch=-1):
        """
        Initializes an ExperimentBuilder object. Such an object takes care of running training and evaluation of a deep net
        on a given dataset. It also takes care of saving per epoch models and automatically inferring the best val model
        to be used for evaluating the test set metrics.
        :param network_model: A pytorch nn.Module which implements a network architecture.
        :param exp_name: The name of the experiment. This is used mainly for keeping track of the experiment and creating and directory structure that will be used to save logs, model parameters and other.
        :param epochs: Total number of epochs to run the experiment
        :param train_data: An object of the DataProvider type. Contains the training set.
        :param val_data: An object of the DataProvider type. Contains the val set.
        :param test_data: An object of the DataProvider type. Contains the test set.
        :param weight_decay_coefficient: A float indicating the weight decay to use with the adam optimizer.
        :param continue_from_epoch: An int indicating whether we'll start from scrach (-1) or whether we'll reload a previously saved model of epoch 'continue_from_epoch' and continue training from there.
        """
        super(ExperimentBuilder, self).__init__()


        self.exp_name = exp_name
        self.flow_name = flow_name
        self.model = network_model

        self.device = device
        self.model.to(self.device)

        self.model.reset_parameters()  # re-initialize network parameters
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        #print('System learnable parameters')
        total_num_parameters = 0
        for name, value in self.named_parameters():
            #print(name, value.shape) # @Todo: possibly look at this again
            total_num_parameters += np.prod(value.shape)

        print('Total number of parameters', total_num_parameters)

        self.optimizer = optimizer
        self.learning_rate_scheduler = scheduler

        # Generate the directory names
        self.exp_path = os.path.join('results', self.exp_name, self.flow_name)
        self.figures_path = os.path.join(self.exp_path, 'figures')
        self.experiment_logs = os.path.join(self.exp_path, 'stats')
        self.experiment_saved_models = os.path.join('saved_models', self.exp_name)

        # Create the folders 
        Path(self.exp_path).mkdir(parents=True, exist_ok=True)
        Path(self.figures_path).mkdir(parents=True, exist_ok=True)
        Path(self.experiment_logs).mkdir(parents=True, exist_ok=True)
        Path(self.experiment_saved_models).mkdir(parents=True, exist_ok=True)

        # Set best models to be at 0 since we are just starting
        self.best_val_model_idx = 0
        self.best_val_model_loss = 1000
        self.epochs = epochs
        self.criterion = error #nn.CrossEntropyLoss().to(self.device)  # send the loss computation to the GPU

        if continue_from_epoch == -2:  # if continue from epoch is -2 then continue from latest saved model
            self.state, self.best_val_model_idx, self.best_val_model_loss = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx='latest')  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = int(self.state['model_epoch'])

        elif continue_from_epoch > -1:  # if continue from epoch is greater than -1 then
            self.state, self.best_val_model_idx, self.best_val_model_loss = self.load_model(
                model_save_dir=self.experiment_saved_models, model_save_name="train_model",
                model_idx=continue_from_epoch)  # reload existing model from epoch and return best val model index
            # and the best val acc of that model
            self.starting_epoch = continue_from_epoch
        else:
            self.state = dict()
            self.starting_epoch = 0

    def get_num_parameters(self):
        total_num_params = 0
        for param in self.parameters():
            total_num_params += np.prod(param.shape)

        return total_num_params

    def run_train_iter(self, inputs_batch, cond_inputs_batch=None):
        self.train()  # sets model to training mode (in case batch normalization or other methods have different procedures for training and evaluation)
        inputs_batch = inputs_batch.float().to(device=self.device) # send data to device as torch tensors
        if cond_inputs_batch is not None:
            cond_inputs_batch = cond_inputs_batch.float().to(device=self.device)
        log_density = self.model.log_prob(inputs_batch, cond_inputs_batch)
        loss = -torch.mean(log_density)
        self.optimizer.zero_grad()  # set all weight grads from previous training iters to 0
        loss.backward()  # backpropagate to compute gradients for current iter loss

        self.optimizer.step()  # update network parameters
        self.learning_rate_scheduler.step()
        return loss.cpu().data.numpy()

    def run_evaluation_iter(self, inputs_batch, cond_inputs_batch=None):
        """
        Receives the inputs and targets for the model and runs an evaluation iterations. Returns loss and accuracy metrics.
        :param x: The inputs to the model. A numpy array of shape batch_size, channels, height, width
        :param y: The targets for the model. A numpy array of shape batch_size, num_classes
        :return: the loss and accuracy for this batch
        """
        self.eval()  # sets the system to validation mode
        inputs_batch = inputs_batch.float().to(device=self.device)  # convert data to pytorch tensors and send to the computation device
        if cond_inputs_batch is not None:
            cond_inputs_batch = cond_inputs_batch.float().to(device=self.device)
        log_density = self.model.log_prob(inputs_batch, cond_inputs_batch)
        loss = -torch.mean(log_density)

        return loss.cpu().data.numpy()

    def save_model(self, model_save_dir, model_save_name, model_idx, best_validation_model_idx,
                   best_validation_model_loss):
        """
        Save the network parameter state and current best val epoch idx and best val accuracy.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :param best_validation_model_idx: The index of the best validation model to be stored for future use.
        :param best_validation_model_loss: The best validation accuracy to be stored for use at test time.
        :param model_save_dir: The directory to store the state at.
        :param state: The dictionary containing the system state.
        """
        self.state['network'] = self.state_dict()  # save network parameter and other variables.
        self.state['best_val_model_idx'] = best_validation_model_idx  # save current best val idx
        self.state['best_val_model_loss'] = best_validation_model_loss  # save current best val acc
        torch.save(self.state, f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(
            model_idx))))  # save state at prespecified filepath

    def load_model(self, model_save_dir, model_save_name, model_idx):
        """
        Load the network parameter state and the best val model idx and best val acc to be compared with the future val accuracies, in order to choose the best val model
        :param model_save_dir: The directory to store the state at.
        :param model_save_name: Name to use to save model without the epoch index
        :param model_idx: The index to save the model with.
        :return: best val idx and best val model acc, also it loads the network state into the system state without returning it
        """
        state = torch.load(f=os.path.join(model_save_dir, "{}_{}".format(model_save_name, str(model_idx))))
        self.load_state_dict(state_dict=state['network'])
        return state, state['best_val_model_idx'], state['best_val_model_loss']

    def run_experiment(self):
        """
        Runs experiment train and evaluation iterations, saving the model and best val model and val model accuracy after each epoch
        :return: The summary current_epoch_losses from starting epoch to total_epochs.
        """
        total_losses = {"train_loss": [], "val_loss": []}  # initialize a dict to keep the per-epoch metrics
        for i, epoch_idx in enumerate(range(self.starting_epoch, self.epochs)):
            epoch_start_time = time.time()
            current_epoch_losses = {"train_loss": [], "val_loss": []}
            self.current_epoch = epoch_idx
            with tqdm.tqdm(total=len(self.train_data)) as pbar_train:  # create a progress bar for training
                for inputs_batch in self.train_data:  # get data batches
                    if isinstance(inputs_batch, list):
                        inputs_batch, cond_inputs_batch = inputs_batch
                    else:
                        cond_inputs_batch = None
                    loss = self.run_train_iter(inputs_batch=inputs_batch, cond_inputs_batch=cond_inputs_batch)  # take a training iter step
                    current_epoch_losses["train_loss"].append(loss)  # add current iter loss to the train loss list
                    pbar_train.update(1)
                    pbar_train.set_description("loss:    {:.4f}".format(loss))

            with tqdm.tqdm(total=len(self.val_data)) as pbar_val:  # create a progress bar for validation
                for inputs_batch in self.val_data:  # get data batches
                    if isinstance(inputs_batch, list):
                        inputs_batch, cond_inputs_batch = inputs_batch
                    else:
                        cond_inputs_batch = None
                    loss = self.run_evaluation_iter(inputs_batch=inputs_batch, cond_inputs_batch=cond_inputs_batch)  # run a validation iter
                    current_epoch_losses["val_loss"].append(loss)  # add current iter loss to val loss list.
                    pbar_val.update(1)  # add 1 step to the progress bar
                    pbar_val.set_description("loss:     {:.4f}".format(loss))
            val_mean_loss = np.mean(current_epoch_losses['val_loss'])
            if val_mean_loss < self.best_val_model_loss:  # if current epoch's mean val acc is greater than the saved best val loss then
                self.best_val_model_loss = val_mean_loss  # set the best val model acc to be current epoch's val loss
                self.best_val_model_idx = epoch_idx  # set the experiment-wise best val idx to be the current epoch's idx
            for key, value in current_epoch_losses.items():
                total_losses[key].append(np.mean(
                    value))  # get mean of all metrics of current epoch metrics dict, to get them ready for storage and output on the terminal.

            save_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv',
                            stats_dict=total_losses, current_epoch=i,
                            continue_from_mode=True if (self.starting_epoch != 0 or i > 0) else False)  # save statistics to stats file.

            # load_statistics(experiment_log_dir=self.experiment_logs, filename='summary.csv') # How to load a csv file if you need to

            out_string = "_".join(
                ["{}_{:.4f}".format(key, np.mean(value)) for key, value in current_epoch_losses.items()])
            # create a string to use to report our epoch metrics
            epoch_elapsed_time = time.time() - epoch_start_time  # calculate time taken for epoch
            epoch_elapsed_time = "{:.4f}".format(epoch_elapsed_time)
            print("Epoch {}: {}, best val epoch: {}, epoch time: {} seconds".format(epoch_idx,
                                                                                    out_string,
                                                                                    self.best_val_model_idx,
                                                                                    epoch_elapsed_time))
            self.state['model_epoch'] = epoch_idx
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx=epoch_idx,
                            best_validation_model_idx=self.best_val_model_idx,
                            best_validation_model_loss=self.best_val_model_loss)
            self.save_model(model_save_dir=self.experiment_saved_models,
                            # save model and best val idx and best val acc, using the model dir, model name and model idx
                            model_save_name="train_model", model_idx='latest',
                            best_validation_model_idx=self.best_val_model_idx,
                            best_validation_model_loss=self.best_val_model_loss)

        print("Generating test set evaluation metrics")
        self.load_model(model_save_dir=self.experiment_saved_models, model_idx=self.best_val_model_idx,
                        # load best validation model
                        model_save_name="train_model")
        current_epoch_losses = {"test_loss": []}  # initialize a statistics dict
        with tqdm.tqdm(total=len(self.test_data)) as pbar_test:  # ini a progress bar
            for inputs_batch in self.test_data:  # sample batch
                if type(inputs_batch) is list:
                    inputs_batch, cond_inputs_batch = inputs_batch
                else:
                    cond_inputs_batch = None
                loss = self.run_evaluation_iter(inputs_batch=inputs_batch, cond_inputs_batch=cond_inputs_batch)  # compute loss and accuracy by running an evaluation step
                current_epoch_losses["test_loss"].append(loss)  # save test loss
                pbar_test.update(1)  # update progress bar status
                pbar_test.set_description(
                    "loss: {:.4f}".format(loss))  # update progress bar string output

        test_losses = {key: [np.mean(value)] for key, value in
                       current_epoch_losses.items()}  # save test set metrics in dict format
        save_statistics(experiment_log_dir=self.experiment_logs, filename='test_summary.csv',
                        # save test set metrics on disk in .csv format
                        stats_dict=test_losses, current_epoch=0, continue_from_mode=False)


        result_dict = load_statistics(self.experiment_logs, filename='summary.csv')

        plot_result_graphs(figures_path=self.figures_path, exp_name=self.exp_name, 
                           stats=result_dict, flow_name=self.flow_name)

        return total_losses, test_losses

