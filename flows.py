from nflows import transforms, distributions, flows, utils
from nflows.nn.nets.resnet import ResidualNet
from utils import gaussian_change_of_var_ND, js_divergence
import torch 
from nflows import transforms, distributions, flows
import torch
import torch.nn as nn


def cop_flow_constructor(n_layers, context_dim, hidden_units=64):

    def create_transform(ii, hidden_units, context_dim):
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features=2, even=(ii % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=hidden_units,
                context_features=context_dim,
                # num_blocks=self.n_blocks_c,
                # dropout_probability=self.dropout_c,
                # use_batch_norm=self.use_batch_norm_c
            ),
            tails='linear',
            tail_bound=16,
            # tails=self.tails,
            # tail_bound=self.tail_bound_c,
            # num_bins=self.n_bins_c,
            # apply_unconditional_transform=self.unconditional_transform
        )
    # Define an invertible transformation.
    transform = transforms.CompositeTransform([
        create_transform(ii, hidden_units, context_dim) for ii in range(n_layers)])

    # Define a base distribution.
    base_distribution = distributions.StandardNormal(shape=[2])

    # Combine into a flow.
    return Cop_Flow(transform=transform, distribution=base_distribution)


def marg_flow_constructor(n_layers):
    def create_transform():
        # return transforms.CompositeTransform([
        #     transforms.MaskedAffineAutoregressiveTransform(features=1, hidden_features=0),
        #     transforms.RandomPermutation(features=1)
        # ])
        # return transforms.PiecewiseRationalQuadraticCouplingTransform(
        #     mask=utils.create_alternating_binary_mask(features=self.dim, even=(ii % 2 == 0)),
        #     transform_net_create_fn=lambda in_features, out_features: nn_.ResidualNet(
        #         in_features=in_features,
        #         out_features=out_features,
        #         hidden_features=self.hidden_units_c,
        #         num_blocks=self.n_blocks_c,
        #         dropout_probability=self.dropout_c,
        #         use_batch_norm=self.use_batch_norm_c
        #     ),
        #     tails=self.tails,
        #     tail_bound=self.tail_bound_c,
        #     num_bins=self.n_bins_c,
        #     apply_unconditional_transform=self.unconditional_transform
        # )
        return transforms.PiecewiseRationalQuadraticCDF(
                shape=[1])

    transform = transforms.CompositeTransform([create_transform() for ii in range(n_layers)])

    # Define a base distribution.
    base_distribution = distributions.StandardNormal(shape=[1])

    # Combine into a flow.
    return Basic_Flow(transform=transform, distribution=base_distribution)


class Basic_Flow(flows.Flow):
    def __init__(self, transform, distribution):
        super().__init__(transform=transform, distribution=distribution)

        self.layer_dict = nn.ModuleDict()

    def reset_parameters(self):
        """
        Re-initialize the network parameters.
        """
        for item in self.layer_dict.children():
            try:
                item.reset_parameters()
            except:
                pass

    def log_pdf_normal(self, inputs, context=None):
        # Here: context normally distributed
        with torch.no_grad():
            if context is None:
                pdf = self.log_prob(inputs)
            else:
                normal_distr = torch.distributions.normal.Normal(0, 1)
                pdf = self.log_prob(inputs, context) + \
                      torch.sum(normal_distr.log_prob(context), axis=1)
            return pdf

    def log_pdf_uniform(self, inputs, context=None):
        with torch.no_grad():
            return gaussian_change_of_var_ND(inputs, self.log_pdf_normal, context=context)

    def jsd(self, true_distribution, context=None, num_samples=100000):
        """Returns JS-Divergence of the predicted distribution and the true distribution
        """
        with torch.no_grad():
            # Get ground truth
            samples_target = true_distribution.sample(num_samples=num_samples, context=context) # @Todo: write distribution function for copulas that can 'sample'
            samples_pred = self.sample_copula(num_samples=num_samples, context=context)

            # Prob X in both distributions
            prob_x_in_p = self.pdf_uniform(inputs=samples_pred, context=context)
            prob_x_in_q = true_distribution.pdf(samples_pred, context=context)

            # Prob Y in both distributions
            prob_y_in_p = self.pdf_uniform(inputs=samples_target, context=context)
            prob_y_in_q = true_distribution.pdf(samples_target, context=context)

            assert torch.min(prob_x_in_p) >= 0
            assert torch.min(prob_x_in_q) >= 0
            assert torch.min(prob_y_in_p) >= 0
            assert torch.min(prob_y_in_q) >= 0

            assert prob_x_in_p.shape == (num_samples,), '{}'.format(prob_x_in_p.shape)
            assert prob_x_in_q.shape == (num_samples,)
            assert prob_y_in_p.shape == (num_samples,)
            assert prob_y_in_q.shape == (num_samples,)

            divergence = js_divergence(prob_x_in_p=prob_x_in_p,
                                       prob_x_in_q=prob_x_in_q,
                                       prob_y_in_p=prob_y_in_p,
                                       prob_y_in_q=prob_y_in_q)

            return divergence


class Cop_Flow(Basic_Flow):
    def __init__(self, transform, distribution):
        super().__init__(transform=transform, distribution=distribution)
        self.norm_distr = torch.distributions.normal.Normal(0, 1)

    def sample(self, num_samples, context=None):
        if context is not None:
            assert context.shape[0] == num_samples, 'Context shape does not match number of samples. \
                                                     Context shape: {}, Number Samples: {}'.format(context.shape[0], num_samples)
        noise = self._distribution.sample(num_samples)
        samples, _ = self._transform.inverse(noise, context=context)
        return samples

    def sample_copula(self, num_samples, context=None):
        samples = self.sample(num_samples, context)
        return self.norm_distr.cdf(samples)