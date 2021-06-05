# Note: These functions are based on functions from https://github.com/CSTR-Edinburgh/mlpractical.
import os
import matplotlib.pyplot as plt
import numpy as np


def collect_experiment_dicts(target_dir, test_flag=False, model_type=''):
    """Collects evaluation metrics from saved files.

    Params:
        target_dir: directory where files live
        test_flag: evaluation on test set
        model_name: model name

    Returns:
        experiment_dicts: experiment dictionaries
    """
    experiment_dicts = dict()
    for subdir, dir, files in os.walk(target_dir):
        for file in files:
            filepath = None
            if not test_flag:
                if file == 'summary_{}.csv'.format(model_type):
                    filepath = os.path.join(subdir, file)

            elif test_flag:
                if file == 'test_summary_{}.csv'.format(model_type):
                    filepath = os.path.join(subdir, file)

            if filepath is not None:

                with open(filepath, 'r') as read_file:
                    lines = read_file.readlines()

                current_experiment_dict = {key: [] for key in lines[0].replace('\n', '').split(',')}
                idx_to_key = {idx: key for idx, key in enumerate(lines[0].replace('\n', '').split(','))}

                for line in lines[1:]:
                    for idx, value in enumerate(line.replace('\n', '').split(',')):
                        current_experiment_dict[idx_to_key[idx]].append(float(value))

                experiment_dicts[subdir.split('/')[-2]] = current_experiment_dict

    return experiment_dicts


def plot_result_graphs(figures_path, model_name, plot_name, stats, model_type=''):
    """Plots training and validation set loss.
    """
    fig_1 = plt.figure(figsize=(8, 4))
    ax_1 = fig_1.add_subplot(111)
    for k in ['train_loss', 'val_loss']:
        item = stats[model_name][k]
        ax_1.plot(np.arange(0, len(item)),
                  item, label='{}'.format(k))

    ax_1.legend(loc=0)
    ax_1.set_ylabel('Loss', fontsize=16)
    ax_1.set_xlabel('Epoch', fontsize=16)

    path = os.path.join(figures_path, '{}_{}_{}_loss_performance.pdf'.format(model_name, model_type, plot_name))
    fig_1.savefig(path, dpi=300, facecolor='w', edgecolor='w',
                  orientation='portrait', format='pdf',
                  transparent=False, bbox_inches='tight', pad_inches=0.1)
