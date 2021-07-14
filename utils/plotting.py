import matplotlib.pyplot as plt
import numpy as np
import os 


def plot_result_graphs(figures_path, exp_name, stats, flow_name=''):
    """Plots training and validation set loss.
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
