import numpy as np
from marg_flow import Marginal_Flow
from cop_flow import Cop_Flow 
from exp_runner import Experiment
from sklearn.model_selection import train_test_split
from utils import set_optimizer_scheduler
from data_provider import DataProvider
from torch.utils.data import DataLoader
import torch
from utils import nll_error


def marginal_transform_1d(inputs: np.ndarray, lr=0.001, weight_decay=0.00001,
                          amsgrad=False, num_epochs=100, batch_size=128, num_workers=4) -> np.ndarray:
    # Transform into data object
    data_train, data_val = train_test_split(inputs, test_size=0.20)
    data_train = DataProvider(data_train)
    loader_train = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                                        # @Todo: take these arguments from somewhere else
    data_val = DataProvider(data_val)
    loader_val = DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                                                        # @Todo: take these arguments from somewhere else

    # Initialize marginal transform
    marg_flow = Marginal_Flow(n_layers=5)
    optimizer, scheduler = set_optimizer_scheduler(marg_flow.flow,
                                                   lr,
                                                   weight_decay,
                                                   amsgrad,
                                                   num_epochs)

    experiment = Experiment(model=marg_flow.flow, optimizer=optimizer, scheduler=scheduler, train_dataset=loader_train,
                            valid_dataset=loader_val, error=nll_error)

    # Train marginal flow
    stats, run_time = experiment.train(num_epochs=num_epochs) # @Todo: adjust stats the way i like it and how it fits to my plotting functions
    print(stats, run_time)
    exit()
    # Evaluate
    # @Todo: implement evaluate (?)
    # @Todo: import save_statistics function, collect_experiment_dicts and plot_results_graph and visualize1D
    # Transform
    outputs = experiment.model.log_prob(inputs)

    return outputs


def marginal_transform(inputs: np.ndarray, lr=0.001, weight_decay=0.00001,
                       amsgrad=False, num_epochs=100, batch_size=128) -> np.ndarray:
    if inputs.shape[1] > 1:
        outputs = torch.empty_like(torch.from_numpy(inputs))
        for dim in range(inputs.shape[1]):
            outputs[:, dim: dim + 1] = marginal_transform_1d(inputs[:, dim: dim+1], lr, weight_decay,
                       amsgrad, num_epochs, batch_size).reshape(-1, 1).detach()
    elif inputs.shape[1] == 1:
        outputs = marginal_transform_1d(inputs, lr, weight_decay,
                       amsgrad, num_epochs, batch_size).reshape(-1, 1).detach()
    else:
        raise ValueError('Invalid input shape.')
    return outputs


def copula_estimator(x_inputs: np.ndarray, y_inputs: np.ndarray,
                     cond_set: np.ndarray, lr=0.001, weight_decay=0.00001,
                       amsgrad=False, num_epochs=100, batch_size=128): # @Todo: find out whether this enters as a tensor or array
    # Transform into data object
    inputs_cond = np.concatenate([x_inputs, y_inputs, cond_set], axis=1)
    data_train, data_val = train_test_split(inputs_cond, test_size=0.20)
    data_train = DataProvider(inputs=data_train[:, :2], cond_inputs=data_train[:, 2:])
    loader_train = DataLoader(data_train, batch_size=64, shuffle=True, num_workers=4)
                                            # @Todo: take these arguments from somewhere else

    data_val = DataProvider(inputs=data_val[:, :2], cond_inputs=data_val[:, 2:])
    loader_val = DataLoader(data_val, batch_size=64, shuffle=True, num_workers=4)
                                            # @Todo: take these arguments from somewhere else

    # Initialize Copula Transform
    cop_flow = Cop_Flow(n_layers=5, context_dim=cond_set.shape[1])
    optimizer, scheduler = set_optimizer_scheduler(cop_flow.flow,
                                                   lr,
                                                   weight_decay,
                                                   amsgrad,
                                                   num_epochs)
    experiment = Experiment(cop_flow.flow, optimizer, scheduler, loader_train, loader_val)

    # Train marginal flow
    stats, keys, run_time = experiment.train(num_epochs=num_epochs)

    # Evaluate
    # @Todo: implement evaluate (?): save logged info somewhere, save experiment stats, save plots

    # Transform
    inputs = torch.cat([x_inputs, y_inputs], axis=1)
    outputs = experiment.model.log_prob(inputs.float(), cond_set.float())

    # Transform to uniform
    normal_distr = torch.distributions.normal.Normal(0, 1)
    outputs = normal_distr.cdf(outputs)  # @Todo: are these outputs needed?

    return outputs, experiment.model  # @Todo: recheck whether this model is then trained...


def mi_estimator(cop_flow, obs_n=1000, obs_m=100) -> float:
    # @Todo: possibly also add ranges
    # @Todo: think about format of mutual information

    ww = torch.FloatTensor(obs_m, 5).uniform_(0, 1)
    ww = torch.distributions.normal.Normal(0, 1).icdf(ww)

    cop_samples = cop_flow._sample(num_samples=obs_n, context=ww) # @Todo: make this run in samples
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
                      cond_set: np.ndarray, num_epochs=100) -> bool:
    # @Todo: add flag for already transformed input, maybe even make every input a tuple

    x_uni = marginal_transform(x_input, num_epochs=num_epochs)
    y_uni = marginal_transform(y_input, num_epochs=num_epochs)
    cond_uni = marginal_transform(cond_set, num_epochs=num_epochs)

    copula_samples, cop_flow = copula_estimator(x_uni, y_uni, cond_uni, num_epochs=num_epochs)

    mutual_information = mi_estimator(cop_flow)

    result = hypothesis_test(mutual_information, threshold=0.95)

    return result


if __name__ == '__main__':
    # @Todo: set seeds

    # Get inputs
    obs = 10000
    epochs = 10
    x = np.random.uniform(size=(obs, 1))
    y = np.random.uniform(size=(obs, 1))
    z = np.random.uniform(size=(obs, 5))
    #
    print(copula_indep_test(x, y, z, num_epochs=epochs))
