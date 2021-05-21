import numpy as np
from marg_flow import Marginal_Flow
from cop_flow import Cop_Flow
from exp_runner import Experiment
from sklearn.model_selection import train_test_split
from utils import set_optimizer_scheduler
from data_provider import DataProvider


def marginal_transform_1D(inputs: np.ndarray, lr=0.001, weight_decay=0.00001,
                       amsgrad=False, num_epochs=100, batch_size=128) -> np.ndarray:
    # Transform into data object
    data_train, data_val = train_test_split(inputs, test_size=0.20)
    data_train = DataProvider(data_train, batch_size)
    data_val = DataProvider(data_val, batch_size)

    # Initialize marginal transform
    marg_flow = Marginal_Flow(n_layers=5)
    optimizer, scheduler = set_optimizer_scheduler(marg_flow.flow,
                                                   lr,
                                                   weight_decay,
                                                   amsgrad,
                                                   num_epochs)

    experiment = Experiment(marg_flow.flow, optimizer, scheduler, data_train, data_val)

    # Train marginal flow
    experiment.train(num_epochs=num_epochs)

    # Evaluate
    # @Todo: implement evaluate (?)

    # Transform
    outputs = experiment.model.log_prob(inputs)

    return outputs


def marginal_transform(inputs: np.ndarray, lr=0.001, weight_decay=0.00001,
                       amsgrad=False, num_epochs=100, batch_size=128) -> np.ndarray:
    if inputs.shape[1] > 1:
        outputs = np.empty_like(inputs)
        for dim in range(inputs.shape[1]):
            outputs[:, dim: dim+1] = marginal_transform_1D(inputs[:, dim: dim+1], lr, weight_decay,
                       amsgrad, num_epochs, batch_size).reshape(-1, 1).detach()
    elif inputs.shape[1] == 1:
        outputs = marginal_transform_1D(inputs, lr, weight_decay,
                       amsgrad, num_epochs, batch_size).reshape(-1, 1).detach()
    else:
        raise ValueError('Invalid input shape.')
    return outputs


def copula_estimator(x_inputs: np.ndarray, y_inputs: np.ndarray,
                     cond_set: np.ndarray, lr=0.001, weight_decay=0.00001,
                       amsgrad=False, num_epochs=100, batch_size=128) -> np.ndarray:
    # Transform into data object
    inputs_cond = np.concatenate([x_inputs, y_inputs, cond_set], axis=1)
    data_train, data_val = train_test_split(inputs_cond, test_size=0.20)
    data_train = DataProvider(inputs=data_train[:, :2], batch_size=batch_size, cond_inputs=data_train[:, 2:])
    data_val = DataProvider(inputs=data_val[:, :2], batch_size=batch_size, cond_inputs=data_val[:, 2:])

    # Initialize Copula Transform
    cop_flow = Cop_Flow(n_layers=5)
    optimizer, scheduler = set_optimizer_scheduler(cop_flow.flow,
                                                   lr,
                                                   weight_decay,
                                                   amsgrad,
                                                   num_epochs)
    experiment = Experiment(cop_flow.flow, optimizer, scheduler, data_train, data_val)

    # Train marginal flow
    experiment.train(num_epochs=num_epochs)

    # Evaluate
    # @Todo: implement evaluate (?)

    # Transform
    inputs = np.concatenate([x_inputs, y_inputs], axis=1)
    outputs = experiment.model.log_prob(inputs, cond_set)

    return outputs


def mi_estimator(cond_copula: np.ndarray) -> float:
    # @Todo: possibly also add ranges
    # @Todo: think about format of mutual information
    raise NotImplementedError
    # return mutual_information


def hypothesis_test(mutual_information: float) -> bool:
    raise NotImplementedError
    # return result


def copula_indep_test(x_input: np.ndarray, y_input: np.ndarray,
                      cond_set: np.ndarray) -> bool:

    x_uni = marginal_transform(x_input)
    exit()
    y_uni = marginal_transform(y_input)
    cond_uni = marginal_transform(cond_set)

    cond_copula = copula_estimator(x_uni, y_uni, cond_uni)
    raise NotImplementedError

    mutual_information = mi_estimator(cond_copula)

    result = hypothesis_test(mutual_information)

    return result


if __name__ == '__main__':
    # @Todo: set seeds

    # Get inputs
    obs = 100
    x = np.random.uniform(size=(obs, 1))
    y = np.random.uniform(size=(obs, 1))
    z = np.random.uniform(size=(obs, 5))
    #
    print(copula_indep_test(x, y, z))
