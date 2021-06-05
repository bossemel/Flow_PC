import torch
import numpy as np
import scipy.stats


def jsd_eval_1d(args, model, test_dict,
                obs=1000, plotname='jsd_test_marginal',
                cm_flow=False):
    """Calculate pointwise JS-Divergence for the predicted marginal distribution.

    Params:
        marginal: marginal distribution
        args: passed arguments
        model: used model
        test_dict: test_dict for evaluation metrics
        obs: number of observation to sample from
        plotname: name of the plot
        cm_flow: whether model is part of cm flow
        marginal_num: which marginal is used

    Returns:
        test_dict: test_dict with evaluation metrics
    """
    with torch.no_grad():
        # Get distributions # @Todo: change this
        marginal_distr = datasets.distributions.Marginals(args.marginal, obs, mu_=args.mu, var_=args.var,
                                                          alpha_=args.alpha, low_=args.low, high_=args.high)
        samples = marginal_distr.sampler()

        # Get Grid
        grid = np.linspace(np.min(samples), np.max(samples), obs).reshape(-1, 1)

        # Prob vector pred
        args.obs = obs
        prob_vector_x = np.exp(model._forward(torch.tensor(grid, device=torch.device(args.device)).float())
                               .cpu().numpy())

        # Prob vector target
        pred_distr_y = scipy.stats.gaussian_kde(samples.T)
        prob_vector_y = pred_distr_y(grid.T).T

        assert np.min(prob_vector_x) >= 0
        assert np.min(prob_vector_y) >= 0

        # Calculate JS Divergence
        divergence = js_divergence_grid(prob_vector_x, prob_vector_y)
        print('JS divergence: ', divergence)

        if cm_flow:
            jsd_name = plotname + '_' + str(cm_flow)
            test_dict[jsd_name] = divergence
        else:
            test_dict[plotname] = divergence
        return test_dict