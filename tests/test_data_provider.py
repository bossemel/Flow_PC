from data_provider import Copula_Distr, Distribution, Joint_Distr, Marginal_Distr
import scipy.stats
import numpy as np
num_samples = 10000

def test_copula_distribution():
    def get_samples_pdf(cop_type, theta):
        copula_distr = Copula_Distr(cop_type, theta=theta)
        samples = copula_distr.sample(num_samples)
        pdf = copula_distr.pdf(samples)
        assert (np.min(samples) > 0) & (np.max(samples) < 1)
        assert (np.min(pdf) > 0)

    get_samples_pdf('clayton', 2)
    get_samples_pdf('frank', 5)
    get_samples_pdf('gumbel', 5)


def test_marginal_distribution():
    def get_samples_pdf(marg_distr, num_samples, mean_gates=(-0.05, 0.05), get_pdf=True, check_mean=False):
        samples = marg_distr.sample(num_samples)
        if check_mean:
            assert (np.mean(samples) > mean_gates[0]) & (np.mean(samples) < mean_gates[0])
        if get_pdf:
            pdf = marg_distr.pdf(samples)
            assert (np.min(pdf) > 0)

    marg_distr = Marginal_Distr('gaussian', mu=0, var=1)
    get_samples_pdf(marg_distr, num_samples=num_samples)

    marg_distr = Marginal_Distr('uniform', low=0, high=1)
    get_samples_pdf(marg_distr, num_samples=num_samples, mean_gates=(0.45, 0.55))

    marg_distr = Marginal_Distr('lognormal', mu=0, var=1)
    get_samples_pdf(marg_distr, num_samples=num_samples, check_mean=False)

    marg_distr = Marginal_Distr('gamma', alpha=5)
    get_samples_pdf(marg_distr, num_samples=num_samples, check_mean=False)

    num_samples_ = 1000

    marg_distr = Marginal_Distr('mix_gamma', mu=0, var=1)
    get_samples_pdf(marg_distr, num_samples=num_samples_, check_mean=False, get_pdf=False)

    marg_distr = Marginal_Distr('mix_lognormal', mu=0, var=1)
    get_samples_pdf(marg_distr, num_samples=num_samples_, check_mean=False, get_pdf=False)

    marg_distr = Marginal_Distr('mix_gaussgamma', mu=0, var=1)
    get_samples_pdf(marg_distr, num_samples=num_samples_, check_mean=False, get_pdf=False)

def test_joint_distribution():
    joint_distr = Joint_Distr('clayton', 'gaussian', 'gaussian', theta=2, mu=0, var=1)
    samples = joint_distr.sample(num_samples)
    assert (np.mean(samples) > -0.05) & (np.mean(samples) < 0.05)

def test_distribution():
    distr_obj = scipy.stats.norm(0, 1)
    samples = Distribution(distr_obj).sample(num_samples)
    assert (np.mean(samples) < 0.05) & (np.mean(samples) > -0.05)
    # @Todo: put true distributions here