# @Todo: write test function that trains and evaluates conditional copula flow for conditional copula inputs
from cond_indep_test import marginal_estimator
from options import TrainOptions
from utils import create_folders, HiddenPrints
import torch
import numpy as np
import json
import random
import os 
from utils.load_and_save import save_statistics
import scipy.stats
from statsmodels.distributions.empirical_distribution import ECDF
from utils import create_folders
from data_provider import split_data_marginal
from options import TrainOptions
from eval.plots import visualize1d
eps = 1e-10


def ecdf_transform(data_train, data_test):
    norm_distr = scipy.stats.norm()
    ecdf = ECDF(data_train.reshape(-1,))
    uniform_samples = ecdf(data_test.reshape(-1,))
    uniform_samples[uniform_samples == 0] = eps
    uniform_samples[uniform_samples == 1] = 1 - eps
    gaussian_samples = norm_distr.ppf(uniform_samples)
    return -np.mean(norm_distr.logpdf(gaussian_samples))

def exp_marg_transform():

    # Training settings
    args = TrainOptions().parse()   # get training options
    args.exp_name = 'exp_marg_flow'

    # Create Folders
    create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.cuda.current_device()

    # Set Seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Get inputs
    obs = 10000
    #args.epochs = 1
    inputs = np.random.standard_normal(size=(obs, 1))

    # Transform into data object
    data_train, data_val, data_test, loader_train, loader_val, loader_test = split_data_marginal(inputs, batch_size=128, num_workers=0, return_datasets=True)

    # Run experiment
    experiment, experiment_metrics, test_metrics = marginal_estimator(loader_train, loader_val, loader_test, args.exp_name, args.device, args.lr, args.weight_decay,
                          args.amsgrad, args.epochs, args.batch_size, args.num_workers, 0)

    # Plot results
    visualize1d(experiment.model, device=args.device, path=experiment.figures_path, true_samples=data_train, obs=1000, name='marg_flow')

    # Transform
    # with torch.no_grad():
    #     inputs = torch.from_numpy(inputs).float().to(args.device)
    #     outputs = experiment.model.transform_to_noise(inputs).cpu().numpy()
        # plt.clf()
        # plt.hist(outputs)
        # plt.savefig('results/hist_norm')

    experiment_logs = os.path.join('results', args.exp_name, 'mf_0', 'stats')

    # Comparison to empirical CDF Transform:
    ecdf_nll = ecdf_transform(data_train, data_test)
    test_metrics['ecdf_nll'] = [ecdf_nll]

    print('Flow NLL: {}, ECDF NLL: {}'.format(test_metrics['test_loss'][0], ecdf_nll))
    save_statistics(experiment_logs, 'test_summary.csv', test_metrics, current_epoch=0, continue_from_mode=False)



def random_search(loader_train, loader_val, loader_test, device, experiment_logs, iterations, epochs): # @Todo: change this, don't change args but make as input directly
    results_dict = {}
    tested_combinations = []
    best_loss = 1000
    ii = 0
    while ii < iterations:
        epochs = epochs
        num_flow_layers_DDSF = np.random.choice(range(1, 8))
        num_hid_layers_DDSF = np.random.choice(range(1, 8))
        dimh_DDSF = 2**np.random.choice(range(1, 7))
        num_ds_dim = 2**np.random.choice(range(1, 7))
        num_ds_layers = np.random.choice(range(1, 8))
        lr = 1 / 10**np.random.choice(range(2, 6))
        weight_decay = 1 / 10**(np.random.choice(range(2, 15)))
        clip_grad_norm = np.random.choice([True, False])
        amsgrad = np.random.choice([True, False])
        clip_m = np.random.choice(range(1, 6))
        batch_size = 2**np.random.choice(4, 9)

        current_hyperparams = (num_flow_layers_DDSF,
                               num_hid_layers_DDSF,
                               dimh_DDSF,
                               num_ds_dim,
                               num_ds_layers,
                               weight_decay,
                               lr,
                               clip_grad_norm,
                               amsgrad,
                               clip_m)
        if current_hyperparams not in tested_combinations:
            print('Num. Flow Layers: {}, Num. Hidden Layers: {}, Num. Hidden Units: {},\
                Num. Sigm. Units: {}, Num. Sigm. Layers: {},\
                Weight Decay: {}, Learning Rate: {}, Clipping: {}, amsgrad: {}'.format(num_flow_layers_DDSF,
                                                                                       num_hid_layers_DDSF,
                                                                                       dimh_DDSF,
                                                                                       num_ds_dim,
                                                                                       num_ds_layers,
                                                                                       weight_decay,
                                                                                       lr,
                                                                                       clip_grad_norm,
                                                                                       amsgrad,
                                                                                       clip_m))
            with HiddenPrints():
                __, experiment_metrics, __ = marginal_estimator(loader_train, 
                                                                loader_val, 
                                                                loader_test, 
                                                                'random_search_marg', 
                                                                device, 
                                                                lr, 
                                                                weight_decay,
                                                                amsgrad, 
                                                                epochs, 
                                                                batch_size, 
                                                                num_workers=0, 
                                                                variable_num=0)
            results_dict[current_hyperparams] = experiment_metrics['val_loss']
            with open(os.path.join(experiment_logs, 'random_search.txt'), 'w') as ff:
                ff.write(str(results_dict))
            if experiment_metrics['val_loss'][0] < best_loss:
                best_loss = experiment_metrics['val_loss']
                best_hyperparams = current_hyperparams
                best_dict = experiment_metrics
            tested_combinations.append(current_hyperparams)
            ii += 1

    print('Random search complete for {}'.format('Marginal Flow'))
    print('Best hyperparams: {}'.format(best_hyperparams))
    print('Lowest Val Loss: {}'.format(best_loss))
    with open(os.path.join(experiment_logs, 'random_search.txt'), 'a') as ff:
        ff.write('Best hyperparams: ' + str(best_hyperparams) + 'Lowest Val Loss: ' + str(best_loss))



if __name__ == '__main__':
    #exp_marg_transform()

    # Training settings
    args = TrainOptions().parse()   # get training options
    args.exp_name = 'exp_marg_flow'

    # Create Folders
    create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.cuda.current_device()

    # Set Seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Get inputs
    obs = 10000
    #args.epochs = 1
    inputs = np.random.standard_normal(size=(obs, 1))
    experiment_logs = os.path.join('results', args.exp_name, 'mf_0', 'stats')

    # Transform into data object
    data_train, data_val, data_test, loader_train, loader_val, loader_test = split_data_marginal(inputs, batch_size=128, num_workers=0, return_datasets=True)

    random_search(loader_train, loader_val, loader_test, args.device, experiment_logs, iterations=200, epochs=50)
    # # Training settings
    # args = TrainOptions().parse()   # get training options

    # # Create Folders
    # args.exp_path = os.path.join('results', args.exp_name)
    # args.figures_path = os.path.join(args.exp_path, args.figures_path)
    # args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    # args.experiment_saved_models = os.path.join(args.experiment_saved_models, args.exp_name)
    # Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    # Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    # Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)
    # Path(args.experiment_saved_models).mkdir(parents=True, exist_ok=True)

    # with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
    #     json.dump(args.__dict__, f, indent=2)

    # # Cuda settings
    # args.cuda = not args.no_cuda and torch.cuda.is_available()
    # args.device = torch.device("cuda:0" if args.cuda else "cpu")
    # args.conditional_copula = False

    # # Set Seed
    # np.random.seed(args.random_seed)
    # torch.manual_seed(args.random_seed)
    # random.seed(args.random_seed)
    # if args.cuda:
    #     torch.cuda.manual_seed(args.random_seed)

    # # Set up data loader
    # dataset, data_loaders = load_data(args)  # @Todo: replace this

    # if args.random_search:
    #     random_search(args)
    # else:
    #     # Train model
    #     model, best_dict, test_dict = train_and_plot(args,
    #                                                  data_loaders)  # Marginal transform 1 D 

