import numpy as np
from flows import Cop_Flow_Constructor, Marg_Flow_Constructor
from exp_runner import ExperimentBuilder
from utils import set_optimizer_scheduler
import torch
from utils import nll_error, create_folders, split_data_marginal, split_data_copula
from options import TrainOptions
import os
import json
import random
import scipy.stats
eps = 1e-10

def marginal_transform_1d(inputs: np.ndarray, exp_name, device, lr=0.001, weight_decay=0.00001,
                          amsgrad=False, num_epochs=100, batch_size=128, num_workers=12, use_gpu=True, variable_num=0) -> np.ndarray:
    # Transform into data object
    loader_train, loader_val, loader_test = split_data_marginal(inputs, batch_size, num_workers=num_workers)

    # Initialize marginal transform
    marg_flow = Marg_Flow_Constructor(n_layers=5)
    optimizer, scheduler = set_optimizer_scheduler(marg_flow.flow,
                                                   lr,
                                                   weight_decay,
                                                   amsgrad,
                                                   num_epochs)

    experiment = ExperimentBuilder(network_model=marg_flow.flow,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   error=nll_error,
                                   exp_name=exp_name,
                                   flow_name= 'mf_' + str(variable_num),
                                   num_epochs=num_epochs,
                                   use_gpu=torch.cuda.is_available(),
                                   train_data=loader_train,
                                   val_data=loader_val,
                                   test_data=loader_test)  # build an experiment object


    # Train marginal flow
    experiment_metrics, test_metrics = experiment.run_experiment()  # run experiment and return experiment metrics

    # Transform
    inputs = torch.from_numpy(inputs).float().to(device)
    outputs = experiment.model.log_prob(inputs)

        # # test_dict = jsd_eval_marginal(args=args,
        # #                           model=model,
        # #                           test_dict=test_dict)
        #
        # visualize1d(model=self.model,
        #             epoch=best_dict['best_validation_epoch'],
        #             args=args,
        #             best_val=True,
        #             name=model_name)
        # # Perform test evaluation
        # test_dict = test(args=args,
        #                  epoch=best_dict['best_validation_epoch'],
        #                  model=model,
        #                  loader=data_loaders['test_loader'],
        #                  device=args.device,
        #                  test_dict=test_dict,
        #                  model_name=model_name,
        #                  disable_tqdm=disable_tqdm)
            # visualize1d(model=model,
            #             epoch=best_dict['best_validation_epoch'],
            #             args=args,
            #             best_val=True,
            #             name=model_name)

    return outputs


def marginal_transform(inputs: np.ndarray, exp_name, device, lr=0.001, weight_decay=0.00001,
                       amsgrad=False, num_epochs=100, batch_size=128) -> np.ndarray:
    if inputs.shape[1] > 1:
        outputs = torch.empty_like(torch.from_numpy(inputs)).to(device).detach()
        for dim in range(inputs.shape[1]):
            outputs[:, dim: dim + 1] = marginal_transform_1d(inputs=inputs[:, dim: dim+1],
                                                             exp_name=exp_name,
                                                             device=device,
                                                             lr=lr, weight_decay=weight_decay,
                                                             amsgrad=amsgrad,
                                                             num_epochs=num_epochs,
                                                             batch_size=batch_size,
                                                             variable_num=dim).reshape(-1, 1).detach()
    elif inputs.shape[1] == 1:
        outputs = marginal_transform_1d(inputs=inputs,  exp_name=exp_name,
                                        device=device, lr=lr, weight_decay=weight_decay, amsgrad=amsgrad,
                                        num_epochs=num_epochs, batch_size=batch_size).reshape(-1, 1).detach()
    else:
        raise ValueError('Invalid input shape.')
    return outputs


def copula_estimator(x_inputs: torch.Tensor, y_inputs: torch.Tensor,
                     cond_set: torch.Tensor, exp_name, device, lr=0.001, weight_decay=0.00001,
                       amsgrad=False, num_epochs=100, batch_size=128, num_workers=12): # @Todo: find out whether this enters as a tensor or array
    # Transform into data object
    inputs_cond = np.concatenate([x_inputs.cpu().numpy(), y_inputs.cpu().numpy(), cond_set.cpu().numpy()], axis=1)
    loader_train, loader_val, loader_test = split_data_copula(inputs_cond, batch_size, num_workers)

    # Initialize Copula Transform
    cop_flow = Cop_Flow_Constructor(n_layers=5, context_dim=cond_set.shape[1])
    optimizer, scheduler = set_optimizer_scheduler(cop_flow.flow,
                                                   lr,
                                                   weight_decay,
                                                   amsgrad,
                                                   num_epochs)

    experiment = ExperimentBuilder(network_model=cop_flow.flow,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   error=nll_error,
                                   exp_name=exp_name,
                                   flow_name= 'cf',
                                   num_epochs=num_epochs,
                                   use_gpu=torch.cuda.is_available(),
                                   train_data=loader_train,
                                   val_data=loader_val,
                                   test_data=loader_test)  # build an experiment object


    # Train marginal flow
    experiment_metrics, test_metrics = experiment.run_experiment()  # run experiment and return experiment metrics

    # # Transform
    # inputs = torch.cat([x_inputs, y_inputs], axis=1)
    # outputs = experiment.model.log_prob(inputs.float().to(device), cond_set.float().to(device))

    # # Transform to uniform
    # normal_distr = torch.distributions.normal.Normal(0, 1)
    # #outputs = normal_distr.cdf(outputs)  # @Todo: are these outputs needed?

    return experiment.model  # @Todo: recheck whether this model is then trained...


def mi_estimator(cop_flow, device, obs_n=20, obs_m=10) -> float:
    ww = torch.FloatTensor(obs_m, 5).uniform_(0, 1)
    ww = torch.distributions.normal.Normal(0, 1).icdf(ww) # @Todo: make this normal directly? why not?

    log_density = torch.empty((ww.shape[0], obs_m))
    for mm in range(obs_m):
        # noise = cop_flow._distribution.sample(ww.shape[0])
        # cop_samples, _ = cop_flow._transform.inverse(noise, context=ww.to(device))
        cop_samples = cop_flow.sample_copula(num_samples=ww.shape[0], context=ww.to(device))
        norm_distr = torch.distributions.normal.Normal(0, 1)
        log_density[:, mm] = cop_flow.log_pdf_uniform(norm_distr.cdf(cop_samples), norm_distr.cdf(ww).to(device)) # @Todo: triple check if this is correct
    
    
    mi = torch.mean(log_density) # @Todo: For this to work, we need ***uniform*** copula density, meaning we need to transwer this density to a true copula!

    return mi.cpu().numpy()


def hypothesis_test(mutual_information: float, threshold: float = 0.05) -> bool: # @Todo: something like num obs?
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
                      cond_set: np.ndarray, exp_name, device, num_epochs=100, num_runs=50, batch_size=64) -> bool:
    x_uni = marginal_transform(x_input, exp_name, device=device, num_epochs=num_epochs, batch_size=batch_size)
    y_uni = marginal_transform(y_input, exp_name, device=device, num_epochs=num_epochs, batch_size=batch_size)
    cond_uni = marginal_transform(cond_set, exp_name, device=device, num_epochs=num_epochs, batch_size=batch_size)

    cop_flow = copula_estimator(x_uni, y_uni, cond_uni, exp_name=exp_name, device=device, num_epochs=num_epochs, batch_size=batch_size)

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
    args.device = torch.device("cuda:0" if use_cuda else "cpu")

    # Set Seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)
    if use_cuda:
        torch.cuda.manual_seed(args.random_seed)

    # Get inputs
    obs = 50
    # args.epochs = 1
    x = np.random.uniform(size=(obs, 1))
    y = np.random.uniform(size=(obs, 1))
    z = np.random.uniform(size=(obs, 5))

    #
    print(copula_indep_test(x, y, z, exp_name=args.exp_name, 
                            device=args.device, num_epochs=args.epochs, batch_size=args.batch_size))
