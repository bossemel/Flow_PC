from nflows import transforms, distributions, flows


class Marginal_Flow:

    def __init__(self, n_layers):
        self.n_layers = n_layers
        # @Todo: add other options
        # Define an invertible transformation.
        transform = transforms.CompositeTransform([
            self.create_transform() for ii in range(self.n_layers)])

        # Define a base distribution.
        base_distribution = distributions.StandardNormal(shape=[1])

        # Combine into a flow.
        self.flow = flows.Flow(transform=transform, distribution=base_distribution)

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