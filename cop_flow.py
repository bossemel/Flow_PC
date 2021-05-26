from nflows import transforms, distributions, flows, utils
import torch.nn as nn
from nflows.nn.nets.resnet import ResidualNet
import torch


class Cop_Flow:

    def __init__(self, n_layers, context_dim, hidden_units=64):
        self.n_layers = n_layers
        self.hidden_units = hidden_units
        self.context_dim = context_dim
        # @Todo: add other options
        # @Todo: set dimension of condition set somewhere?
        # Define an invertible transformation.
        transform = transforms.CompositeTransform([
            self.create_transform(ii) for ii in range(self.n_layers)])

        # Define a base distribution.
        base_distribution = distributions.StandardNormal(shape=[2])

        # Combine into a flow.
        self.flow = flows.Flow(transform=transform, distribution=base_distribution)

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
