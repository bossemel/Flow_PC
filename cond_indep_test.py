import numpy as np
from marg_flow import Marginal_Flow
from cop_flow import Cop_Flow 
from exp_runner import ExperimentBuilder
from utils import set_optimizer_scheduler
import torch
from utils import nll_error, create_folders, split_data_marginal, split_data_copula
from options import TrainOptions
import os
import json
import random


def marginal_transform_1d(inputs: np.ndarray, exp_name, experiment_logs_dir, device, lr=0.001, weight_decay=0.00001,
                          amsgrad=False, num_epochs=100, batch_size=128, num_workers=12, use_gpu=True) -> np.ndarray:
    # Transform into data object
    loader_train, loader_val, loader_test = split_data_marginal(inputs, batch_size, num_workers=num_workers)

    # Initialize marginal transform
    marg_flow = Marginal_Flow(n_layers=5)
    optimizer, scheduler = set_optimizer_scheduler(marg_flow.flow,
                                                   lr,
                                                   weight_decay,
                                                   amsgrad,
                                                   num_epochs)

    experiment = ExperimentBuilder(network_model=marg_flow.flow,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   error=nll_error,
                                   exp_name=exp_name,
                                   num_epochs=num_epochs,
                                   use_gpu=torch.cuda.is_available(),
                                   train_data=loader_train,
                                   val_data=loader_val,
                                   test_data=loader_test)  # build an experiment object


    # Train marginal flow
    experiment_metrics, test_metrics = experiment.run_experiment()  # run experiment and return experiment metrics

    # Transform
    inputs = torch.from_numpy(inputs).float().to(device)
    outputs = experiment.model.log_prob(inputs)

    return outputs


def marginal_transform(inputs: np.ndarray, exp_name, experiment_logs_dir, device, lr=0.001, weight_decay=0.00001,
                       amsgrad=False, num_epochs=100, batch_size=128) -> np.ndarray:
    if inputs.shape[1] > 1:
        outputs = torch.empty_like(torch.from_numpy(inputs)).to(device).detach()
        for dim in range(inputs.shape[1]):
            outputs[:, dim: dim + 1] = marginal_transform_1d(inputs=inputs[:, dim: dim+1],
                                                             exp_name=exp_name,
                                                             experiment_logs_dir=experiment_logs_dir,
                                                             device=device,
                                                             lr=lr, weight_decay=weight_decay,
                                                             amsgrad=amsgrad,
                                                             num_epochs=num_epochs,
                                                             batch_size=batch_size).reshape(-1, 1).detach()
    elif inputs.shape[1] == 1:
        outputs = marginal_transform_1d(inputs=inputs,  exp_name=exp_name, experiment_logs_dir=experiment_logs_dir,
                                        device=device, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad,
                                        num_epochs=num_epochs, batch_size=batch_size).reshape(-1, 1).detach()
    else:
        raise ValueError('Invalid input shape.')
    return outputs


def copula_estimator(x_inputs: np.ndarray, y_inputs: np.ndarray,
                     cond_set: np.ndarray, exp_name, device, lr=0.001, weight_decay=0.00001,
                       amsgrad=False, num_epochs=100, batch_size=128, num_workers=12): # @Todo: find out whether this enters as a tensor or array
    # Transform into data object
    inputs_cond = np.concatenate([x_inputs.cpu().numpy(), y_inputs.cpu().numpy(), cond_set.cpu().numpy()], axis=1)
    loader_train, loader_val, loader_test = split_data_copula(inputs_cond, batch_size, num_workers)

    # Initialize Copula Transform
    cop_flow = Cop_Flow(n_layers=5, context_dim=cond_set.shape[1])
    optimizer, scheduler = set_optimizer_scheduler(cop_flow.flow,
                                                   lr,
                                                   weight_decay,
                                                   amsgrad,
                                                   num_epochs)

    experiment = ExperimentBuilder(network_model=cop_flow.flow,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   error=nll_error,
                                   exp_name=exp_name,
                                   num_epochs=num_epochs,
                                   use_gpu=torch.cuda.is_available(),
                                   train_data=loader_train,
                                   val_data=loader_val,
                                   test_data=loader_test)  # build an experiment object


    # Train marginal flow
    experiment_metrics, test_metrics = experiment.run_experiment()  # run experiment and return experiment metrics

    # Transform
    inputs = torch.cat([x_inputs, y_inputs], axis=1)
    outputs = experiment.model.log_prob(inputs.float(), cond_set.float())

    # Transform to uniform
    normal_distr = torch.distributions.normal.Normal(0, 1)
    outputs = normal_distr.cdf(outputs)  # @Todo: are these outputs needed?

    return outputs, experiment.model  # @Todo: recheck whether this model is then trained...


def mi_estimator(cop_flow, device, obs_n=1000, obs_m=100) -> float:
    # @Todo: possibly also add ranges
    # @Todo: think about format of mutual information

    ww = torch.FloatTensor(obs_m, 5).uniform_(0, 1)
    ww = torch.distributions.normal.Normal(0, 1).icdf(ww)

    cop_samples = cop_flow._sample(num_samples=obs_n, context=ww.to(device)) # @Todo: make this run in samples
    cop_samples = torch.distributions.normal.Normal(0, 1).cdf(cop_samples)
    # @Todo: remove 0 from  output

    # @Todo: calculate conditional mutual information
    mi = torch.mean(torch.log(cop_samples))
    # @Todo: think about a way to add error bars
    print(mi)
    return float(mi)


def hypothesis_test(mutual_information: float, threshold: float) -> bool: # @Todo: something like num obs?
    raise NotImplementedError
    # return result


def copula_indep_test(x_input: np.ndarray, y_input: np.ndarray,
                      cond_set: np.ndarray, exp_name, experiment_logs_dir, device, num_epochs=100) -> bool:
    x_uni = marginal_transform(x_input, exp_name, experiment_logs_dir, device=device, num_epochs=num_epochs)
    y_uni = marginal_transform(y_input, exp_name, experiment_logs_dir, device=device, num_epochs=num_epochs)
    cond_uni = marginal_transform(cond_set, exp_name, experiment_logs_dir, device=device, num_epochs=num_epochs)

    copula_samples, cop_flow = copula_estimator(x_uni, y_uni, cond_uni, exp_name=exp_name, device=device, num_epochs=num_epochs)

    mutual_information = mi_estimator(cop_flow, device=device)

    result = hypothesis_test(mutual_information, threshold=0.95)

    return result


if __name__ == '__main__':

    # Training settings
    args = TrainOptions().parse()   # get training options

    # Create Folders
    create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.device("cuda:0" if use_cuda else "cpu")

    # Set Seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Get inputs
    obs = 10000
    epochs = 1
    x = np.random.uniform(size=(obs, 1))
    y = np.random.uniform(size=(obs, 1))
    z = np.random.uniform(size=(obs, 5))
    #
    print(copula_indep_test(x, y, z, exp_name=args.exp_name, experiment_logs_dir=args.experiment_logs,
                            device=args.device, num_epochs=epochs))
