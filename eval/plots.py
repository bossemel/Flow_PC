import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import seaborn as sns
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout


def visualize1d(model, true_samples: np.ndarray, device: str, path: str, obs: int =1000, name: str ='') -> None:
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
    path = os.path.join(path, '{}_bestval.pdf'.format(name))
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved {} to {}'.format(path.split('/')[-1], path))


def histogram(samples: np.ndarray, path: str, var_name: str, plt_name: str) -> None:
    """Visualizes the true and predicted marginals.

    Params:
        marginal: kind of marginal distribution
        model: DDSF model
        epoch: epoch to use
        args: passed input arguments
        obs: number of observations to samples
    """
    fig = plt.figure(figsize=(8, 6))
    sns.distplot(samples, kde=False, norm_hist=True) #, color='royalblue')
    plt.xlabel(var_name, fontsize=20)
    plt.ylabel('relative frequency', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    fig.tight_layout()
    path = os.path.join(path, '{}.pdf'.format(plt_name))
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved {} to {}'.format(path.split('/')[-1], path))


def visualize_joint(data: np.ndarray, figures_path: str, name: str, axis_1_name: str =None, axis_2_name: str =None) -> None:
    """Visualize 2D distribution as a seaborn jointplot.
    Params:
        data: 2D distribution
        figures_path: path to save the figure
        name: name of the figure
        axis_1_name: name of the first axis
        axis_2_name: name of the second axis
    """
    if axis_1_name is None:
        axis_1_name = 'X1'
    if axis_2_name is None:
        axis_2_name = 'X2'

    fig = plt.figure()
    fig = sns.jointplot(x=data[:, 0], y=data[:, 1], kind='hex')
    fig.set_axis_labels(axis_1_name, axis_2_name, fontsize=16)
    path = os.path.join(figures_path, name + '.pdf')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print('Saved {} to {}'.format(path.split('/')[-1], path))

def plot_result_graphs(figures_path: str, exp_name: str, stats: dict, flow_name: str ='') -> None:
    """Plots training and validation set loss.
    Params:
        figures_path: path to save the figure
        exp_name: name of the experiment
        stats: dictionary with evaluation statistics
        flow_name: name of the estimated flow
    """
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['train_loss', 'val_loss']:
        item = stats[k]
        ax_1.plot(np.arange(0, len(item)),
                  item, label='{}'.format(k))

    ax_1.legend(loc=0)
    ax_1.set_ylabel('Loss', fontsize=16)
    ax_1.set_xlabel('Epoch', fontsize=16)

    path = os.path.join(figures_path, '{}_{}_loss_performance.pdf'.format(exp_name, flow_name))
    fig_1.savefig(path, dpi=300, facecolor='w', edgecolor='w',
                  orientation='portrait', format='pdf',
                  transparent=False, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print('Saved {} to {}'.format(path.split('/')[-1], path))

def plot_graph(graph, path):
    """
    Plot the graph G with nodes and edges at positions pos.
    Params:
        graph: networkx graph
        path: path to save the figure
    """
    plt.figure(figsize=(10,10))
    pos = graphviz_layout(graph, prog="circo")
    nx.draw(graph, pos, with_labels=True, node_size=3000, font_size=20, 
            alpha=0.9, width=2)
    plt.savefig(path, format='pdf', dpi=300)
    plt.close()
    print('Saved {} to {}'.format(path.split('/')[-1], path))