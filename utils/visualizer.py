import seaborn as sns
import matplotlib.pyplot as plt
import os


def visualize_joint(data, figures_path, name, axis_1_name=None, axis_2_name=None):
    """Visualize 2D distribution as a seaborn jointplot.
    """
    if axis_1_name is None:
        axis_1_name = 'X1'
    if axis_2_name is None:
        axis_2_name = 'X2'

    fig = plt.figure()
    fig = sns.jointplot(data[:, 0], data[:, 1], kind='hex', stat_func=None)
    fig.set_axis_labels(axis_1_name, axis_2_name, fontsize=16)
    fig.savefig(os.path.join(figures_path, name + '.pdf'), dpi=300, bbox_inches='tight')
