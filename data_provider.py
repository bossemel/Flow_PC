import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.stats
import scipy
import pynverse
import torch
import os
import pyvinecopulib as pv
import random
from pathlib import Path
import sklearn
from utils.copula_sampling import sample_clayton, sample_frank, sample_gumbel, copula_pdf
eps = 1e-07


def marginal_transform(inputs, marginal, mu=None, var=None, alpha=None):
    """Transforms the uniform copula marginals into a different distribution.

    Params:
        inputs: copula samples vector
        marginal: desired marginal distributions
        args: passed input arguments

    Returns:
        inputs: transformed samples vector
    """
    if marginal == 'gaussian':
        norm = scipy.stats.norm(loc=mu, scale=var)
        inputs = norm.ppf(inputs)
    elif marginal == 'uniform':
        return inputs
    elif marginal == 'lognormal':
        lognorm = scipy.stats.lognorm(s=0.5, loc=mu, scale=var)
        inputs = lognorm.ppf(inputs)
    elif marginal == 'gamma':
        gamma = scipy.stats.gamma(alpha)
        inputs = gamma.ppf(inputs)
    elif marginal == 'gmm':
        def distr_1(xx): return scipy.stats.norm.cdf(xx, loc=mu - 2, scale=var * 2)
        def distr_2(xx): return scipy.stats.norm.cdf(xx, loc=mu + 2, scale=var / 2)
        def distr_3(xx): return scipy.stats.norm.cdf(xx, loc=mu, scale=var / 4)
    elif marginal == 'mix_gamma':
        def distr_1(xx): return scipy.stats.gamma.cdf(xx, 1)
        def distr_2(xx): return scipy.stats.gamma.cdf(xx, 5)
        def distr_3(xx): return scipy.stats.gamma.cdf(xx, 2)
    elif marginal == 'mix_lognormal':
        def distr_1(xx): return scipy.stats.lognorm.cdf(xx, s=0.1, loc=mu - 2, scale=var * 2)
        def distr_2(xx): return scipy.stats.lognorm.cdf(xx, s=0.9, loc=mu + 2, scale=var / 2)
        def distr_3(xx): return scipy.stats.lognorm.cdf(xx, s=0.5, loc=mu, scale=var)
    elif marginal == 'mix_gauss_gamma':
        def distr_1(xx): return scipy.stats.norm.cdf(xx, loc=mu, scale=var / 5)
        def distr_2(xx): return scipy.stats.gamma.cdf(xx, alpha)
        def distr_3(xx): return scipy.stats.gamma.cdf(xx, alpha * 5)
    if marginal in ['gmm', 'mix_gamma', 'mix_lognormal', 'mix_gauss_gamma']:
        inverse_cdf = pynverse.inversefunc(lambda xx: 0.4 * distr_1(xx) + 0.4 * distr_2(xx) + 0.2 * distr_3(xx))
        inputs = inverse_cdf(np.longdouble(inputs))
    return inputs


class Joint_Distr:
    """Class for bivariate samples given a copula correlation and individual marginals.
    """

    def __init__(self, copula, marginal_1_, marginal_2_, theta, mu=None, var=None, alpha=None):

        self.mu = mu
        self.var = var
        self.alpha = alpha
        self.theta = theta

        self.copula = copula
        self.marginal_1 = marginal_1_
        self.marginal_2 = marginal_2_

    def sample(self, obs_=None):
        """Returns copula samples.
        """
        copuladistr = CopulaDistr(self.copula,
                                    self.theta,
                                    obs_)
        samples = copuladistr.sample(obs_=obs_, transform=False)
        xx = marginal_transform(inputs=samples,
                                marginal=self.marginal_1,
                                mu=self.mu,
                                var=self.var,
                                alpha=self.alpha)
        return xx


class Marginal_Distr:
    """Class for univariate samples
    """
    def __init__(self, marginal, mu=None, var=None, alpha=None, low=None, high=None):
        self.marginal = marginal
        self.mu = mu
        self.var = var
        self.alpha = alpha
        self.low = low
        self.high = high

    def sample(self, num_samples):
        """Returns marginal samples.
        """
        dataset = scipy.stats.uniform.rvs(size=num_samples)
        dataset = marginal_transform(dataset, self.marginal, mu=self.mu, var=self.var, alpha=self.alpha)
        return dataset.reshape(-1, 1)

    def pdf(self, inputs):
        if self.marginal == 'gaussian':
            pdf_samples = scipy.stats.norm.pdf(inputs,
                                               loc=self.mu,
                                               scale=self.var)
        elif self.marginal == 'uniform':
            assert hasattr(self, 'low'), 'Please specify lower bound a for %r distribution' % self.marginal
            assert hasattr(self, 'high'), 'Please specify upper bound b for %r distribution' % self.marginal

            pdf_samples = scipy.stats.uniform.pdf(inputs,
                                                  loc=self.low,
                                                  scale=self.high)
        elif self.marginal == 'gamma':
            assert self.alpha is not None, 'Please specify %r for %r distribution' % self.marginal

            pdf_samples = scipy.stats.gamma.pdf(inputs,
                                                a=self.alpha)

        elif self.marginal == 'lognormal':
            pdf_samples = scipy.stats.lognorm.pdf(inputs,
                                                  s=0.5,
                                                  loc=self.mu,
                                                  scale=self.var)
        else:
            raise ValueError('Unknown marginal')

        return pdf_samples


class CopulaDistr:
    def __init__(self, copula, theta, transform=True):

        self.copula = copula
        self.theta = theta
        self.transform = transform

    def sample(self, obs_=None, transform=None):
        """Produce obs_ samples of 2-dimensional Copula density distribution
        """
        # Following Copula definitions from
        # https://pydoc.net/copulalib/1.1.0/copulalib.copulalib/
        # Conditional Distribution Method:
        # clayton copula
        if self.copula == 'clayton':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % self.copula
            uu, vv = sample_clayton(obs_, self.theta)

        # frank copula
        elif self.copula == 'frank':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % self.copula
            uu, vv = sample_frank(obs_, self.theta)

        # gumbel copula
        elif self.copula == 'gumbel':
            assert hasattr(self, 'theta'), 'Please specify theta for %r copula' % self.copula
            uu, vv = sample_gumbel(obs_, self.theta)

        # gumbel copula
        elif self.copula == 'independent':
            xx = scipy.stats.uniform.rvs(size=(obs_, 2))

        else:
            raise ValueError('Unknown copula')

        if self.copula in ['clayton', 'frank', 'gumbel']:
            xx = np.concatenate([uu.reshape(-1, 1), vv.reshape(-1, 1)], axis=1)

        assert xx.all() > 0 & xx.all() < 1

        # Apply inverse Gaussian
        if transform:
            norm = scipy.stats.norm()
            xx = norm.ppf(xx)

        return xx

    def pdf(self, xx):
        uu = xx[:, 0]
        vv = xx[:, 1]
        return copula_pdf(self.copula, self.theta, uu, vv)


def save_dataset_2D(copula, marginal_1, marginal_2, theta, obs_, mu, var, alpha):
    dataset = Joint_Distr(copula, marginal_1, marginal_2, theta,
                          mu=mu, var=var, alpha=alpha)
    samples = dataset.sample(obs_)

    data_train, data_val = train_test_split(samples, test_size=0.2)
    data_val, data_test = train_test_split(samples, test_size=0.5)

    np.save(os.path.join(os.path.join('datasets', 'joint_data'), '2D_{}_{}_{}_trn'.format(copula, marginal_1, marginal_2)), data_train, )
    np.save(os.path.join(os.path.join('datasets', 'joint_data'), '2D_{}_{}_{}_val'.format(copula, marginal_1, marginal_2)), data_val)
    np.save(os.path.join(os.path.join('datasets', 'joint_data'), '2D_{}_{}_{}_tst'.format(copula, marginal_1, marginal_2)), data_test)


def save_dataset_4D(mix, copula='clayton', marginal='gamma', obs_=10000):
    samples = mutivariate_copula(mix, marginal=marginal, disable_marginal=False)
    np.save(os.path.join(os.path.join('datasets', 'joint_data'), '4D_{}_{}_mix{}'.format(copula, marginal, mix)), samples)


class Distribution():
    def __init__(self, distribution):
        self.distribution = distribution 
        self.sample = distribution.rvs


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


def split_data_marginal(inputs, batch_size, num_workers=12, return_datasets=False):
    # Transform into data object
    data_train, data_val = train_test_split(inputs, test_size=0.2)
    data_val, data_test = train_test_split(data_val, test_size=0.5)

    # Normalize
    scaler = sklearn.preprocessing.StandardScaler()
    data_train = scaler.fit_transform(data_train)
    data_val = scaler.transform(data_val)
    data_test = scaler.transform(data_test)

    provider_train = DataProvider(data_train)
    loader_train = torch.utils.data.DataLoader(provider_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
            
    provider_val = DataProvider(data_val)
    loader_val = torch.utils.data.DataLoader(provider_val, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    provider_test = DataProvider(data_test)
    loader_test = torch.utils.data.DataLoader(provider_test, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    if return_datasets:
        return data_train, data_val, data_test, loader_train, loader_val, loader_test
    return loader_train, loader_val, loader_test


class TorchStandardScaler:
  def fit(self, x):
    self.mean = x.mean(0, keepdim=True)
    self.std = x.std(0, unbiased=False, keepdim=True)
  def transform(self, x):
    x -= self.mean
    x /= (self.std + 1e-7)
    return x


def split_data_copula(x_inputs, y_inputs, cond_set, batch_size, num_workers):
    inputs_cond = torch.cat([x_inputs, y_inputs, cond_set], axis=1)

    data_train, data_val = train_test_split(inputs_cond, test_size=0.20)
    data_val, data_test = train_test_split(data_val, test_size=0.50)

    # Normalize
    scaler = TorchStandardScaler()
    scaler.fit(data_train)
    data_train = scaler.transform(data_train)
    data_val = scaler.transform(data_val)
    data_test = scaler.transform(data_test)
    
    data_train = DataProvider(inputs=data_train[:, :2], cond_inputs=data_train[:, 2:])
    loader_train = torch.utils.data.DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=0 if x_inputs.is_cuda else num_workers)


    data_val = DataProvider(inputs=data_val[:, :2], cond_inputs=data_val[:, 2:])
    loader_val = torch.utils.data.DataLoader(data_val, batch_size=batch_size, shuffle=True, num_workers=0 if x_inputs.is_cuda else num_workers)


    data_test = DataProvider(inputs=data_test[:, :2], cond_inputs=data_test[:, 2:])
    loader_test = torch.utils.data.DataLoader(data_test, batch_size=batch_size, shuffle=True, num_workers=0 if x_inputs.is_cuda else num_workers)

    return loader_train, loader_val, loader_test


def mutivariate_copula(mix, copula='clayton', marginal='gamma', obs_=10000, disable_marginal=False):
    if mix is False:
        if copula == 'clayton':
            pair_copula = pv.BicopFamily.clayton
            theta = 2
        elif copula == 'frank':
            pair_copula = pv.BicopFamily.frank
            theta = 5
        elif copula == 'gumbel':
            pair_copula = pv.BicopFamily.gumbel
            theta = 5
        else:
            raise ValueError('Unknown copula type.')

        # Specify pair-copulas
        bicop = pv.Bicop(family=pair_copula, parameters=[theta])
        pcs = [[bicop, bicop, bicop], [bicop, bicop], [bicop]]
    else:
        bicop_1 = pv.Bicop(family=pv.BicopFamily.gumbel, parameters=[5])
        bicop_2 = pv.Bicop(family=pv.BicopFamily.clayton, parameters=[2])
        bicop_3 = pv.Bicop(family=pv.BicopFamily.frank, parameters=[5])
        pcs = [[bicop_1, bicop_2, bicop_3], [bicop_1, bicop_2], [bicop_1]]

    # Specify R-vine matrix
    mat = np.array([[1, 1, 1, 1], [2, 2, 2, 0], [3, 3, 0, 0], [4, 0, 0, 0]])

    # Set-up a vine copula
    copula = pv.Vinecop(matrix=mat, pair_copulas=pcs)
    copulasamples = copula.simulate(n=obs_)
    if not disable_marginal:
        for dim in range(copulasamples.shape[1]):
            copulasamples[:, dim] = marginal_transform(copulasamples[:, dim], marginal=marginal,
                                                                  mu=mu, var=var, alpha=alpha)
    assert not np.isnan(np.sum(copulasamples)), '{}'.format(copulasamples[np.isnan(copulasamples)])
    return torch.from_numpy(copulasamples)


if __name__ == '__main__':
    path = os.path.join('datasets', 'joint_data')
    Path(path).mkdir(parents=True, exist_ok=True)
    copulalist = ['clayton', 'frank', 'gumbel', 'independent']
    marginal_1_list = ['gaussian', 'uniform', 'gamma', 'lognormal', 'gmm', 'mix_gamma', 'mix_lognormal',
                       'mix_gauss_gamma']
    marginal_2_list = ['gaussian', 'uniform', 'gamma', 'lognormal', 'gmm', 'mix_gamma', 'mix_lognormal',
                       'mix_gauss_gamma']
    alpha = 10
    mu = 0
    var = 1
    obs = 10000
    seed = 4

    # Set Seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)

    for copula in copulalist:
        for marginal_1 in marginal_1_list:
            # for marginal_2 in marginal_2_list:
            if copula == 'clayton':
                theta = 2
            else:
                theta = 5
            print('Creating 2D dataset for {} copula with {} marginal..'.format(copula, marginal_1))
            save_dataset_2D(copula, marginal_1, marginal_1, theta, obs_=obs, mu=mu, var=var, alpha=alpha)
            if copula != 'independent':
                print('Creating 4D dataset for {} copula with {} marginal..'.format(copula, marginal_1))
                save_dataset_4D(False, copula, marginal_1, obs)

    for marginal_1 in marginal_1_list:
        print('Creating 4D dataset for {} copula with mixed marginals..'.format(marginal_1))
        save_dataset_4D(True, marginal=marginal_1, obs_=obs)
