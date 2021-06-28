import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns


def visualize1d(model, true_samples, device, path, obs=1000, name=''):
    """Visualizes the true and predicted marginals.

    Params:
        marginal: kind of marginal distribution
        model: DDSF model
        epoch: epoch to use
        args: passed input arguments
        obs: number of observations to samples
    """
    fig = plt.figure(figsize=(8, 6))

    #data = marginal_distr.sampler()
    sns.distplot(true_samples, bins=100, kde=False, label='Test Samples', norm_hist=True, color='orange')

    xx = torch.linspace(np.min(true_samples), np.max(true_samples), obs).reshape(-1, 1)
    with torch.no_grad():
        zz = np.exp(model.log_prob(xx.to(device)).data.detach().cpu().numpy())
    plt.plot(xx.numpy(), zz, label='Predicted PDF', color='royalblue', linewidth=3.0)
    plt.xlabel('x', fontsize=20)
    plt.ylabel('Probability', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    fig.legend(bbox_to_anchor=(0, 0, 0.97, 0.97), fontsize=20)
    fig.tight_layout()

    fig.savefig(os.path.join(path, '{}_bestval.pdf'.format(name)),
                    dpi=300, bbox_inches='tight')



def plot_copula():
    raise NotImplementedError

def plot_marginal():
    raise NotImplementedError


def visualize_joint(data, figures_path, name, axis_1_name=None, axis_2_name=None):
    """Visualize 2D distribution as a seaborn jointplot.
    """
    if axis_1_name is None:
        axis_1_name = 'X1'
    if axis_2_name is None:
        axis_2_name = 'X2'

    fig = plt.figure()
    fig = sns.jointplot(data[:, 0], data[:, 1], kind='hex')
    fig.set_axis_labels(axis_1_name, axis_2_name, fontsize=16)
    fig.savefig(os.path.join(figures_path, name + '.pdf'), dpi=300, bbox_inches='tight')
