import torch.optim as optim
import torch
import os
from pathlib import Path
import sys
eps = 1e-7


def set_optimizer_scheduler(model, lr, weight_decay, amsgrad, epochs):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay, amsgrad=amsgrad)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    return optimizer, scheduler

def nll_error(log_density):
    return -torch.mean(log_density)

def create_folders(args):
    args.exp_path = os.path.join('results', args.exp_name)
    args.figures_path = os.path.join(args.exp_path, 'figures')
    args.experiment_logs = os.path.join(args.exp_path, 'result_outputs')
    args.experiment_saved_models = os.path.join('saved_models', args.exp_name)
    Path(args.exp_path).mkdir(parents=True, exist_ok=True)
    Path(args.figures_path).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_logs).mkdir(parents=True, exist_ok=True)
    Path(args.experiment_saved_models).mkdir(parents=True, exist_ok=True)


class HiddenPrints:
    """Hide Prints (for grid search).
    """
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

# class Fixed_sampler(flows.Flow):
#     def __init__(self, dependent=False):
#         super().__init__(transform=None, distribution=None)
#         self.dependent = dependent 

#     def sample(self, num_samples, context=None):
#         if not self.dependent:
#             return torch.from_numpy(scipy.stats.norm.rvs(size=(num_samples, 2)))

#     def log_prob(self, inputs, context=None):
#         if not self.dependent:
#             mult_norm = scipy.stats.multivariate_normal(mean=[0,0], cov=[[1,0],[0,1]])
#             return torch.log(torch.from_numpy(mult_norm.pdf(inputs)))

#     #def _distribution(self):



def gaussian_change_of_var_ND(inputs: torch.Tensor, original_log_pdf, context=None):
    inputs[inputs == 0] = eps
    inputs[inputs == 1] = 1 - eps
    normal_distr = torch.distributions.normal.Normal(0, 1)
    assert torch.max(inputs) < 1, '{}'.format(torch.max(inputs))
    assert torch.min(inputs) > 0, '{}'.format(torch.min(inputs))

    recast_inputs = normal_distr.icdf(inputs).float()

    if context is not None:
        context[context == 0] = eps
        context[context == 1] = 1 - eps
        assert torch.max(context) < 1, '{}'.format(torch.max(context))
        assert torch.min(context) > 0, '{}'.format(torch.min(context))
        recast_context = normal_distr.icdf(context).float()
        copy_recast_inputs = recast_inputs.detach()
        original_joint = original_log_pdf(copy_recast_inputs, context=recast_context)
    else:
        copy_recast_inputs = recast_inputs.detach()
        original_joint = original_log_pdf(copy_recast_inputs)

    if context is not None:
        recast_inputs = torch.cat([recast_inputs, recast_context], axis=1)
    second_dim = recast_inputs.shape[1] if len(recast_inputs.shape) == 2 else 1

    if second_dim >= 2:
        determinant = normal_distr.log_prob(recast_inputs).sum(axis=1) # torch.exp(normal_distr.log_prob(recast_inputs).sum(axis=1))
    else:
        determinant = normal_distr.log_prob(recast_inputs) # torch.exp(normal_distr.log_prob(recast_inputs))

    output = original_joint - determinant
    #(output) >= 0, '{}'.format(torch.min(output))
    return output

