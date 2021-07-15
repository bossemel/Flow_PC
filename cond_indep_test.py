import numpy as np
from flows import Basic_Flow, cop_flow_constructor, marg_flow_constructor
from exp_runner import ExperimentBuilder
from utils import set_optimizer_scheduler
import torch
from utils import nll_error, create_folders, set_seeds
from options import TrainOptions
from data_provider import split_data_marginal,  split_data_copula
import os
import json
import random
import scipy.stats
import tqdm 
eps = 1e-10


def marginal_estimator(loader_train: torch.utils.data.DataLoader, loader_val: torch.utils.data.DataLoader, 
                       loader_test: torch.utils.data.DataLoader, exp_name: str, device: str,
                       epochs: int =50, variable_num: int =0, disable_tqdm: bool =False, **kwargs) -> tuple:
    # Initialize marginal transform
    marg_flow = marg_flow_constructor(**kwargs) 

    optimizer, scheduler = set_optimizer_scheduler(marg_flow, 
                                                   lr=kwargs['lr'], 
                                                   weight_decay=kwargs['weight_decay'], 
                                                   amsgrad=kwargs['amsgrad'], 
                                                   epochs=epochs)

    experiment = ExperimentBuilder(network_model=marg_flow,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   error=nll_error,
                                   exp_name=exp_name,
                                   flow_name= 'mf_' + str(variable_num),
                                   epochs=epochs,
                                   train_data=loader_train,
                                   val_data=loader_val,
                                   test_data=loader_test,
                                   device=device,
                                   disable_tqdm=disable_tqdm)  # build an experiment object


    # Train marginal flow
    experiment_metrics, test_metrics = experiment.run_experiment()  # run experiment and return experiment metrics
    return experiment, experiment_metrics, test_metrics


def marginal_transform_1d(inputs: np.ndarray, exp_name: str, device: str, epochs: int =100, batch_size: int =128, 
                          num_workers: int =0, variable_num: int =0, **kwargs) -> np.ndarray:
    # Transform into data object
    loader_train, loader_val, loader_test = split_data_marginal(inputs, batch_size, num_workers=num_workers)

    experiment, __, __ = marginal_estimator(loader_train=loader_train, 
                                            loader_val=loader_val, 
                                            loader_test=loader_test, 
                                            exp_name=exp_name, 
                                            device=device,
                                            epochs=epochs, 
                                            batch_size=batch_size, 
                                            num_workers=num_workers, 
                                            variable_num=variable_num, 
                                            **kwargs)
    
    # Transform
    inputs = torch.from_numpy(inputs).float().to(device)
    outputs = experiment.model.transform_to_noise(inputs)

    return outputs


def marginal_transform(inputs: np.ndarray, exp_name: str, device: str, epochs: int =100, 
                       batch_size: int =128, num_workers: int =0, **kwargs) -> np.ndarray:
    if inputs.shape[1] > 1:
        outputs = torch.empty_like(torch.from_numpy(inputs)).to(device).detach()
        for dim in range(inputs.shape[1]):
            outputs[:, dim: dim + 1] = marginal_transform_1d(inputs=inputs[:, dim: dim+1],
                                                             exp_name=exp_name,
                                                             device=device,
                                                             epochs=epochs,
                                                             batch_size=batch_size,
                                                             variable_num=dim,
                                                             num_workers=num_workers,
                                                             **kwargs).reshape(-1, 1).detach()
    elif inputs.shape[1] == 1:
        outputs = marginal_transform_1d(inputs=inputs,  exp_name=exp_name,
                                        device=device, epochs=epochs, batch_size=batch_size, **kwargs).reshape(-1, 1).detach()
    else:
        raise ValueError('Invalid input shape.')
    return outputs


def copula_estimator(loader_train: torch.utils.data.DataLoader, loader_val: torch.utils.data.DataLoader, 
                     loader_test: torch.utils.data.DataLoader, cond_set_dim: int, exp_name: str, device: str, epochs: int =100, 
                     disable_tqdm: bool =False, **kwargs) -> tuple:

    # Initialize Copula Transform
    cop_flow = cop_flow_constructor(context_dim=cond_set_dim, **kwargs)
    optimizer, scheduler = set_optimizer_scheduler(cop_flow,
                                                   kwargs['lr'],
                                                   kwargs['weight_decay'],
                                                   kwargs['amsgrad'],
                                                   epochs)

    experiment = ExperimentBuilder(network_model=cop_flow,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   error=nll_error,
                                   exp_name=exp_name,
                                   flow_name= 'cf',
                                   epochs=epochs,
                                   train_data=loader_train,
                                   val_data=loader_val,
                                   test_data=loader_test,
                                   device=device,
                                   disable_tqdm=disable_tqdm)  # build an experiment object


    # Train marginal flow
    experiment_metrics, test_metrics = experiment.run_experiment()  # run experiment and return experiment metrics

    return experiment, experiment_metrics, test_metrics


def mi_estimator(cop_flow: Basic_Flow, device: str, obs_n: int =1000, obs_m: int =1000, cond_set_dim: int =False) -> float:
    if cond_set_dim:
        ww = torch.FloatTensor(obs_m, cond_set_dim).normal_(0, 1)

    log_density = torch.empty((ww.shape[0] if cond_set_dim else obs_n, obs_m))
    for mm in range(obs_m):
        cop_samples = cop_flow.sample_copula(num_samples=ww.shape[0] if cond_set_dim else obs_n, context=ww.to(device) if cond_set_dim else None)
        norm_distr = torch.distributions.normal.Normal(0, 1)
        log_density[:, mm] = cop_flow.log_pdf_uniform(norm_distr.cdf(cop_samples), context=norm_distr.cdf(ww).to(device) if cond_set_dim else None)
    
    mi = torch.mean(log_density)
    return mi.cpu().numpy()


def hypothesis_test(mutual_information: float, threshold: float = 0.05) -> bool:
    statistic, pvalue = scipy.stats.ttest_1samp(mutual_information, 0, axis=0, nan_policy='raise')
    print('Test statistic: {}, P-Value: {}'.format(statistic, pvalue))
    print('Threshold: ', threshold)
    if pvalue > threshold:
        print('MI not significantly different from zero. Sample conditionally independent')
        return True
    elif pvalue <= threshold:
        print('MI significantly different from zero. Samples not conditionally independent.')
        return False
    else:
        print('Invalid test result.')


def mi_loop(cop_flow: Basic_Flow, cond_set_dim: int, num_runs: int, device: str) -> list:
    mi_runs = []
    ii = 0
    with tqdm.tqdm(total=num_runs) as pbar_test:  # ini a progress bar
        while ii < num_runs:
            mi_estimate = mi_estimator(cop_flow, device=device, cond_set_dim=cond_set_dim)
            if not np.isnan(mi_estimate):
                mi_runs.append(mi_estimate)
                ii += 1
            pbar_test.update(1)
    return mi_runs


def copula_indep_test(x_input: np.ndarray, y_input: np.ndarray,
                      cond_set: np.ndarray, exp_name: str, device: str, kwargs_m, kwargs_c, epochs_m: int, epochs_c: int, 
                      num_runs: int=30, batch_size_m: int =128, batch_size_c: int =128, num_workers: int =0) -> bool:
    
    print('Estimating x marginal...')
    x_uni = marginal_transform(x_input, exp_name, device=device, epochs=epochs_m, batch_size=batch_size_m, num_workers=num_workers, **kwargs_m)
    print('Estimating y marginal...')
    y_uni = marginal_transform(y_input, exp_name, device=device, epochs=epochs_m, batch_size=batch_size_m, num_workers=num_workers, **kwargs_m)
    print('Estimating cond set marginals...')
    cond_uni = marginal_transform(cond_set, exp_name, device=device, epochs=epochs_m, batch_size=batch_size_m, num_workers=num_workers, **kwargs_m)

    # Transform into data object
    print('Creating copula dataset..')
    data_train, data_val, data_test, loader_train, loader_val, loader_test = split_data_copula(x_uni, 
                                                                                               y_uni,
                                                                                               cond_uni, 
                                                                                               batch_size=128, 
                                                                                               num_workers=0, 
                                                                                               return_datasets=True)

    print('Estimating copula..')
    experiment, __, __ = copula_estimator(loader_train, loader_val, loader_test, cond_set_dim=cond_uni.shape[-1], 
                                          exp_name=exp_name, device=device, epochs=epochs_c, batch_size=batch_size_c, 
                                          num_workers=num_workers, **kwargs_c)
    cop_flow = experiment.model

    print('Estimating mutual information..')
    with torch.no_grad():
        cop_flow.eval()
        mi_runs = mi_loop(cop_flow, cond_uni.shape[-1], num_runs, device)

    print('Running hypothesis test..')
    result = hypothesis_test(np.array(mi_runs), threshold=0.05)
    return result


if __name__ == '__main__':

    # Training settings
    args = TrainOptions().parse()   # get training options
    # args.experiment_logs = os.path.join('results', args.exp_name, 'mf_0', 'stats')
    args.flow_name = ''    

    # Create Folders
    create_folders(args)
    with open(os.path.join(args.experiment_logs, 'args'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

    # Cuda settings
    use_cuda = torch.cuda.is_available()
    args.device = torch.cuda.current_device()

    # Set Seed
    set_seeds(seed=args.seed, use_cuda=use_cuda)

    # Get inputs
    obs = 50
    x = np.random.uniform(size=(obs, 1))
    y = np.random.uniform(size=(obs, 1))
    z = np.random.uniform(size=(obs, 5))

    # kwargs marginal
    kwargs_m = {'n_layers': args.n_layers_m,
              'lr': args.lr_m,
              'weight_decay': args.weight_decay_m,
              'amsgrad': args.amsgrad_m,
              'n_bins': args.n_bins_m,
              'tail_bound': args.tail_bound_m,
              'hidden_units': args.hidden_units_m,
              'tails': args.tails_m,
              'identity_init': args.identity_init_m}

    # kwargs copula
    kwargs_c = {'n_layers': args.n_layers_c,
              'lr': args.lr_c,
              'weight_decay': args.weight_decay_c,
              'amsgrad': args.amsgrad_c,
              'n_bins': args.n_bins_c,
              'tail_bound': args.tail_bound_c,
              'hidden_units': args.hidden_units_m,
              'tails': args.tails_m,
              'identity_init': args.identity_init_m}
    #
    print(copula_indep_test(x, y, z, exp_name=args.exp_name, 
                            device=args.device, 
                            kwargs_m=kwargs_m,
                            kwargs_c=kwargs_c,
                            epochs_m=args.epochs_m, 
                            epochs_c=args.epochs_c, 
                            batch_size_m=args.batch_size_m, 
                            batch_size_c=args.batch_size_c, 
                            num_workers=args.num_workers))

