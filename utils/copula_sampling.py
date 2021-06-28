import numpy as np
import sys
eps = 1e-7


def sample_clayton(obs_, theta_, uu=None, ww=None, random_seed=None):
    """Sample from clayton copula density

    Params:
        obs_: how many samples to generate
        theta_: clayton copula parameter
        uu, ww: fixed input grid

    Returns:
        uu, vv: samples
    """
    if uu is None:
        if random_seed is None:
            uu = np.random.uniform(size=obs_)
            ww = np.random.uniform(size=obs_)
        else:
            uu = np.random.RandomState(random_seed).uniform(size=obs_)
            ww = np.random.RandomState(random_seed + 1).uniform(size=obs_)

    if theta_ <= -1:
        raise ValueError('the parameter for clayton copula should be more than -1')
    elif theta_ == 0:
        raise ValueError('The parameter for clayton copula should not be 0')

    if theta_ < sys.float_info.epsilon:
        vv = ww
    else:
        vv = uu * (ww ** (-theta_ / (1 + theta_)) - 1 + uu ** theta_) ** (-1 / theta_)
    return uu, vv


def sample_frank(obs_, theta_, uu=None, ww=None, random_seed=None):
    """Sample from frank copula density

    Params:
        obs_: how many samples to generate
        theta_: frank copula parameter
        uu, ww: fixed input grid

    Returns:
        uu, vv: samples
    """
    if uu is None:
        if random_seed is None:
            uu = np.random.uniform(size=obs_)
            ww = np.random.uniform(size=obs_)
        else:
            uu = np.random.RandomState(random_seed).uniform(size=obs_)
            ww = np.random.RandomState(random_seed + 1).uniform(size=obs_)

    if theta_ == 0:
        raise ValueError('The parameter for frank copula should not be 0')

    if abs(theta_) > np.log(sys.float_info.max):
        vv = (uu < 0) + np.sign(theta_) * uu
    elif abs(theta_) > np.sqrt(sys.float_info.epsilon):
        vv = -np.log((np.exp(-theta_ * uu) * (1 - ww) / ww + np.exp(-theta_)) /
                     (1 + np.exp(-theta_ * uu) * (1 - ww) / ww)) / theta_
    else:
        vv = ww
    return uu, vv


def sample_gumbel(obs_, theta_, random_seed=None):
    """Sample from gumbel copula density

    Params:
        obs_: how many samples to generate
        theta_: gumbel copula parameter
        uu, ww: fixed input grid

    Returns:
        uu, vv: samples
    """
    if theta_ <= 1:
        raise ValueError('the parameter for gumbel copula should be greater than 1')
    if theta_ < 1 + sys.float_info.epsilon:
        if random_seed is None:
            uu = np.random.uniform(size=obs_)
            ww = np.random.uniform(size=obs_)
        else:
            uu = np.random.RandomState(random_seed).uniform(size=obs_)
            ww = np.random.RandomState(random_seed + 1).uniform(size=obs_)
    else:
        if random_seed is None:
            u_int = np.random.uniform(size=obs_)
            ww = np.random.uniform(size=obs_)
            w1 = np.random.uniform(size=obs_)
            w2 = np.random.uniform(size=obs_)
        else:
            u_int = np.random.RandomState(random_seed).uniform(size=obs_)
            ww = np.random.RandomState(random_seed + 1).uniform(size=obs_)
            w1 = np.random.RandomState(random_seed + 2).uniform(size=obs_)
            w2 = np.random.RandomState(random_seed + 3).uniform(size=obs_)

        u_int = (u_int - 0.5) * np.pi
        u2 = u_int + np.pi / 2
        ee = -np.log(ww)
        tt = np.cos(u_int - u2 / theta_) / ee
        gamma = (np.sin(u2 / theta_) / tt) ** (1 / theta_) * tt / np.cos(u_int)
        s1 = (-np.log(w1)) ** (1 / theta_) / gamma
        s2 = (-np.log(w2)) ** (1 / theta_) / gamma
        uu = np.array(np.exp(-s1))
        vv = np.array(np.exp(-s2))
    assert not np.isnan(np.sum(uu))
    assert not np.isnan(np.sum(vv))
    assert uu.all() >= 0
    assert vv.all() >= 0
    return uu, vv


def _g(theta_, z):
    r"""Helper function to solve frank copula.
    This functions encapsulates :math:`g(z) = e^{-\theta_ z} - 1` used on frank copulas.
    Argument:
        z: np.ndarray
    Returns:
        np.ndarray
    Source:
        https://github.com/sdv-dev/Copulas/blob/master/copulas/bivariate/clayton.py
    """
    return np.exp(np.multiply(-theta_, z)) - 1


def gumbel_cdf(theta_, uu, vv):
    if theta_ == 1:
        return np.multiply(uu, vv)

    else:
        h = np.power(-np.log(uu), theta_) + np.power(-np.log(vv), theta_)
        h = -np.power(h, 1.0 / theta_)
        cdfs = np.exp(h)
        return cdfs


def copula_pdf(copula_, theta_, uu, vv):
    uu = remove_0_1(uu).numpy().astype('float64')
    vv = remove_0_1(vv).numpy().astype('float64')
    assert np.min(uu) > 0 and np.max(uu) < 1, 'min: {}, max: {}'.format(np.min(uu), np.max(uu))
    assert np.min(vv) > 0 and np.max(vv) < 1, 'min: {}, max: {}'.format(np.min(vv), np.max(vv))

    if copula_ == 'clayton':
        a = (theta_ + 1) * np.power(np.multiply(uu, vv), -(theta_ + 1))
        assert np.isfinite(a.sum()), 'np.multiply(uu, vv): {}, -(theta_ + 1): {}'.format(np.multiply(uu, vv).dtype,
                                                                                         type(-(theta_ + 1)))
        b = np.power(uu, -theta_) + np.power(vv, -theta_) - 1
        c = -(2 * theta_ + 1) / theta_
        pdf = a * np.power(b, c, dtype=np.float64)
        assert np.min(pdf) > 0, 'clayton_{}_{}_b:{} c: {}'.format(np.min(pdf), theta_, b, c)
        return pdf
    if copula_ == 'frank':
        if theta_ == 0:
            return np.multiply(uu, vv)

        else:
            num = np.multiply(np.multiply(-theta_, _g(theta_, 1)), 1 + _g(theta_, np.add(uu, vv)))
            aux = np.multiply(_g(theta_, uu), _g(theta_, vv)) + _g(theta_, 1)
            den = np.power(aux, 2, dtype=np.float64)
            pdf = num / den
            assert np.min(pdf) >= 0, 'frank_{}'.format(np.min(pdf))
            return pdf
    if copula_ == 'gumbel':
        if theta_ == 1:
            return np.multiply(uu, vv)

        else:
            a = np.power(np.multiply(uu, vv), -1, dtype=np.float64)
            tmp = np.power(-np.log(uu), theta_) + np.power(-np.log(vv), theta_)
            b = np.power(tmp, -2 + 2.0 / theta_, dtype=np.float64)

            c = np.power(np.multiply(np.log(uu), np.log(vv)), theta_ - 1)

            d = 1 + (theta_ - 1) * np.power(tmp, -1.0 / theta_, dtype=np.float64)
            pdf = gumbel_cdf(theta_, uu, vv) * a * b * c * d
            assert np.min(pdf) >= 0, 'gumbel_{}'.format(np.min(pdf))
            return pdf
    else:
        raise NotImplementedError


def remove_0_1(array):
    array[array == 0] = eps
    array[array == 1] = 1 - eps
    return array
