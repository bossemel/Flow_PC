from nflows import transforms, distributions, flows, utils
from nflows.nn.nets.resnet import ResidualNet
from utils import gaussian_change_of_var_ND
import torch 
from nflows import transforms, distributions, flows
import torch
import torch.nn as nn


class Cop_Flow_Constructor:
    def __init__(self, n_layers, context_dim, hidden_units=64):
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.context_dim = context_dim

        # Define an invertible transformation.
        transform = transforms.CompositeTransform([
            self.create_transform(ii) for ii in range(self.n_layers)])

        # Define a base distribution.
        base_distribution = distributions.StandardNormal(shape=[2])

        # Combine into a flow.
        self.flow = Cop_Flow(transform=transform, distribution=base_distribution)

    def create_transform(self, ii):
        return transforms.PiecewiseRationalQuadraticCouplingTransform(
            mask=utils.create_alternating_binary_mask(features=2, even=(ii % 2 == 0)),
            transform_net_create_fn=lambda in_features, out_features: ResidualNet(
                in_features=in_features,
                out_features=out_features,
                hidden_features=self.hidden_units,
                context_features=self.context_dim,
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


class Marg_Flow_Constructor:
    def __init__(self, n_layers):
        self.n_layers = n_layers
        # @Todo: add other options
        # Define an invertible transformation.
        transform = transforms.CompositeTransform([
            self.create_transform() for ii in range(self.n_layers)])

        # Define a base distribution.
        base_distribution = distributions.StandardNormal(shape=[1])

        # Combine into a flow.
        self.flow = Basic_Flow(transform=transform, distribution=base_distribution)

    def create_transform(self):
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