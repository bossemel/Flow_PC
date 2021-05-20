import numpy as np
from marg_flow import Marginal_Flow#, Copula_Flow
from exp_runner import Experiment
from sklearn.model_selection import train_test_split
from utils import set_optimizer_scheduler
from data_provider import DataProvider


def marginal_transform(inputs: np.ndarray, lr=0.001, weight_decay=0.00001,
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


def copula_estimator(x_inputs: np.ndarray, y_inputs: np.ndarray,
                     cond_set: np.ndarray) -> np.ndarray:
    raise NotImplementedError

    # Initialize Copula Transform
    # @Todo: implement copula flow class
    cop_flow = Copula_Flow(x_inputs, y_inputs, cond_set)
    experiment = Experiment(cop_flow, x_inputs.shape, cond_set.shape)

    # Create Training and validation sets
    # @Todo: Create Training and validation sets

    # Train marginal flow
    experiment.train(x)

    # Evaluate
    # @Todo: implement evaluate (?)

    # Transform
    outputs = experiment.model.forward(x_inputs, y_inputs, cond_set)

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
    y_uni = marginal_transform(y_input)
    raise NotImplementedError # @Todo: implement marginal transform for the conditioning set as well
    cond_uni = marginal_transform(cond_set)

    cond_copula = copula_estimator(x_uni, y_uni, cond_uni)

    mutual_information = mi_estimator(cond_copula)

    result = hypothesis_test(mutual_information)

    return result


if __name__ == '__main__':
    # @Todo: set seeds

    # Get inputs
    obs = 1000
    x = np.random.uniform(size=(obs, 1))
    y = np.random.uniform(size=(obs, 1))
    z = np.random.uniform(size=(obs, 5))
    #
    print(copula_indep_test(x, y, z))
