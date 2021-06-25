import numpy as np
from flows import cop_flow_constructor, marg_flow_constructor
from exp_runner import ExperimentBuilder
from utils import set_optimizer_scheduler
import torch
from utils import nll_error, create_folders
from options import TrainOptions
from data_provider import split_data_marginal,  split_data_copula
import os
import json
import random
import scipy.stats
eps = 1e-10


def marginal_estimator(loader_train, loader_val, loader_test, exp_name, device,
                          epochs=100, variable_num=0, disable_tqdm=False, **kwargs):
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


def marginal_transform_1d(inputs: np.ndarray, exp_name, device, epochs=100, batch_size=128, num_workers=0, variable_num=0, **kwargs) -> np.ndarray:
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


def marginal_transform(inputs: np.ndarray, exp_name, device, epochs=100, batch_size=128, num_workers=0, **kwargs) -> np.ndarray:
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


def copula_estimator(x_inputs: torch.Tensor, y_inputs: torch.Tensor,
                     cond_set: torch.Tensor, exp_name, device, epochs=100, batch_size=128, num_workers=0, disable_tqdm=False, **kwargs):
    # Transform into data object
    loader_train, loader_val, loader_test = split_data_copula(x_inputs, y_inputs, cond_set, batch_size, num_workers)

    # Initialize Copula Transform
    cop_flow = cop_flow_constructor(context_dim=cond_set.shape[-1], **kwargs)
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

    # # Transform
    # inputs = torch.cat([x_inputs, y_inputs], axis=1)
    # outputs = experiment.model.log_prob(inputs.float().to(device), cond_set.float().to(device))

    # # Transform to uniform
    # normal_distr = torch.distributions.normal.Normal(0, 1)
    # #outputs = normal_distr.cdf(outputs)  # @Todo: are these outputs needed?

    return experiment, experiment_metrics, test_metrics


def mi_estimator(cop_flow, device, obs_n=20, obs_m=10) -> float:
    ww = torch.FloatTensor(obs_m, 5).normal_(0, 1)

    log_density = torch.empty((ww.shape[0], obs_m))
    for mm in range(obs_m):
        # noise = cop_flow._distribution.sample(ww.shape[0])
        # cop_samples, _ = cop_flow._transform.inverse(noise, context=ww.to(device))
        cop_samples = cop_flow.sample_copula(num_samples=ww.shape[0], context=ww.to(device))
        norm_distr = torch.distributions.normal.Normal(0, 1)
        log_density[:, mm] = cop_flow.log_pdf_uniform(norm_distr.cdf(cop_samples), norm_distr.cdf(ww).to(device)) # @Todo: triple check if this is correct
    
    
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


def copula_indep_test(x_input: np.ndarray, y_input: np.ndarray,
                      cond_set: np.ndarray, exp_name, device, kwargs_m, kwargs_c, epochs=100, num_runs=50, batch_size=64, num_workers=0) -> bool:
    x_uni = marginal_transform(x_input, exp_name, device=device, epochs=epochs, batch_size=batch_size, num_workers=num_workers, **kwargs_m)
    y_uni = marginal_transform(y_input, exp_name, device=device, epochs=epochs, batch_size=batch_size, num_workers=num_workers, **kwargs_m)
    cond_uni = marginal_transform(cond_set, exp_name, device=device, epochs=epochs, batch_size=batch_size, num_workers=num_workers, **kwargs_m)

    experiment, __, __ = copula_estimator(x_uni, y_uni, cond_uni, exp_name=exp_name, device=device, epochs=epochs, batch_size=batch_size, num_workers=num_workers, **kwargs_c)
    cop_flow = experiment.model

    with torch.no_grad():
        cop_flow.eval()
        mi_runs = []
        ii = 0
        while ii < num_runs:
            mi_estimate = mi_estimator(cop_flow, device=device)
            if not np.isnan(mi_estimate):
                mi_runs.append(mi_estimate)
                ii += 1


    result = hypothesis_test(np.array(mi_runs), threshold=0.05)

    return result


if __name__ == '__main__':

    # Training settings
    args = TrainOptions().parse()   # get training options

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
    obs = 50
    x = np.random.uniform(size=(obs, 1))
    y = np.random.uniform(size=(obs, 1))
    z = np.random.uniform(size=(obs, 5))

    # kwargs marginal
    kwargs_m = {'n_layers': args.n_layers_c,
              'lr': args.lr_c,
              'weight_decay': args.weight_decay_c,
              'amsgrad': args.amsgrad_c}

    # kwargs copula
    kwargs_c = {'n_layers': args.n_layers_c,
              'lr': args.lr_c,
              'weight_decay': args.weight_decay_c,
              'amsgrad': args.amsgrad_c}
    #
    print(copula_indep_test(x, y, z, exp_name=args.exp_name, 
                            device=args.device, 
                            kwargs_m=kwargs_m,
                            kwargs_c=kwargs_c,
                            epochs=args.epochs, 
                            batch_size=args.batch_size, 
                            num_workers=args.num_workers))
