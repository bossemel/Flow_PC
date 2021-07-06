import argparse


class TrainOptions:
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """
    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def dataset_marginal(self):
        self.parser.add_argument(
            '--marginal', default='uniform',
            choices=['gaussian', 'uniform', 'gamma', 'lognormal', 'gmm', 'mix_gamma',
                     'mix_lognormal', 'mix_gauss_gamma'], help='marginal in first dimension')
        self.parser.add_argument(
            '--alpha', type=float, default=5, help='alpha for gamma distribution')
        self.parser.add_argument(
            '--mu', type=float, default=0, help='mu for marginal gaussian distribution')
        self.parser.add_argument(
            '--var', type=float, default=1, help='var for marginal gaussian distribution')
        self.parser.add_argument(
            '--low', type=float, default=0, help='lower bound for uniform distribution')
        self.parser.add_argument(
            '--high', type=float, default=1, help='upper bound for uniform distirbution')
        self.parser.add_argument(
            '--obs', type=int, default=10000, help='How many data samples to generate')

    def dataset_copula(self):
        self.parser.add_argument(
            '--copula', default='clayton',
            choices=['gaussian', 'tdistr', 'clayton', 'frank', 'gumbel', 'independent'])
        self.parser.add_argument(
            '--theta', type=float, default=2.0, help='theta for copula sampling')

    def optimizer_marg(self):
        self.parser.add_argument(
            '--weight_decay_m', type=float, default=1e-10, help='adam optimizer weight decay')
        self.parser.add_argument(
            '--lr_m', type=float, default=0.01, help='learning rate (default: 0.0001)')
        self.parser.add_argument(
            '--amsgrad_m', action='store_true', default=False)
        self.parser.add_argument(
            '--batch_size_m', type=int, default=128, help='input batch size for training')
        self.parser.add_argument(
            '--epochs_m', type=int, default=50, help='number of epochs to train (default: 100)')
        self.parser.add_argument(
            '--clip_grad_norm_m', action='store_true', default=False, help='number of epochs to train (default: 100)')

    def optimizer_cop(self):
        self.parser.add_argument(
            '--weight_decay_c', type=float, default=0.0001, help='adam optimizer weight decay')
        self.parser.add_argument(
            '--lr_c', type=float, default=0.001, help='learning rate (default: 0.0001)')
        self.parser.add_argument(
            '--amsgrad_c', action='store_false', default=True)
        self.parser.add_argument(
            '--batch_size_c', type=int, default=128, help='input batch size for training')
        self.parser.add_argument(
            '--epochs_c', type=int, default=50, help='number of epochs to train (default: 100)')
        self.parser.add_argument(
            '--clip_grad_norm_c', action='store_false', default=True)
        self.parser.add_argument(
            '--dropout_c', type=float, default=0.15, help='Dropout probability in flow')
        self.parser.add_argument(
            '--unconditional_transform_c', action='store_true', default=False, help='Unconditionally transform identity features')

# @Todo: write the correct help functions
    def marg_flow(self):
        self.parser.add_argument(
            '--n_layers_m', type=int, default=5, help='Number of spline layers in flow')
        self.parser.add_argument(
            '--hidden_units_m', type=int, default=4, help='Number of hidden units in spline layer')
        self.parser.add_argument(
            '--n_bins_m', type=int, default=45, help='Number of bins in piecewise spline transform')
        self.parser.add_argument(
            '--tail_bound_m', type=int, default=32, help='Bounds of spline region')
        self.parser.add_argument(
            '--identity_init_m', action='store_true', default=False, help='adam optimizer weight decay')
        self.parser.add_argument(
            '--tails_m', type=str, default=None, help='Function type outside spline region')

    def cop_flow(self):
        self.parser.add_argument(
            '--n_layers_c', type=int, default=8, help='Number of spline layers in flow')
        self.parser.add_argument(
            '--hidden_units_c', type=int, default=32, help='Number of hidden units in spline layer')
        self.parser.add_argument(
            '--n_blocks_c', type=int, default=2, help='Number of residual blocks in each spline layer')
        self.parser.add_argument(
            '--n_bins_c', type=int, default=40, help='Number of bins in piecewise spline transform')
        self.parser.add_argument(
            '--tail_bound_c', type=int, default=32, help='Bounds of spline region')
        self.parser.add_argument(
            '--batch_norm_c', type=int, default=True, help='Bounds of spline region')
        self.parser.add_argument(
            '--identity_init_c', action='store_false', default=True, help='adam optimizer weight decay')
        self.parser.add_argument(
            '--tails_c', type=str, default='linear', help='Function type outside spline region')

    def initialize(self):
        # Training settings
        self.parser = argparse.ArgumentParser(description='PyTorch Flows')

        self.parser.add_argument(
            '--exp_name', default='default_name', help='experiment name')
        self.parser.add_argument(
            '--seed', type=int, default=4, help='random seed')
        self.parser.add_argument(
            '--num_workers', type=int, default=0, help='number of workers in the data loader (default: 0)')


        # Dataset
        self.dataset_marginal()
        self.dataset_copula()


        # Optim Options
        self.optimizer_marg()
        self.optimizer_cop()

        # Flow Options
        self.marg_flow()
        self.cop_flow()

        self.parser.add_argument(
            '--num_hid_layers', type=int, default=4, help='adam optimizer weight decay')



        # Architecture
        # parser.add_argument(
        #     '--conditional_copula', action='store_true', help='estimates the conditional copula')
        # parser.add_argument(
        #     '--cop_flow', default='NSF', choices=['NSF', 'RealNVP'], help='which flow type to use for copula flow')
        # parser.add_argument(
        #     '--marg_flow', default='DDSF', choices=['NSF', 'DDSF'], help='which flow type to use for marginal flow')
        # parser.add_argument(
        #     '--use_ecdf', action='store_true')
        #
        # # Training options
        # parser.add_argument(
        #     '--batch_size', type=int, default=128, help='input batch size for training')

        # parser.add_argument(
        #     '--epochs_m', type=int, default=100, help='number of epochs to train (default: 100)')
        # parser.add_argument(
        #     '--lr_c', type=float, default=0.001, help='learning rate (default: 0.0001)')
        # parser.add_argument(
        #     '--lr_m', type=float, default=0.00001, help='learning rate (default: 0.0001)')
        # parser.add_argument(
        #     '--no-cuda', action='store_true', default=False, help='disables CUDA training')
        # parser.add_argument(
        #     '--random_seed', type=int, default=4, help='random seed')
        # parser.add_argument(
        #     '--clip_grad_norm_c', action='store_true', help='whether to clip gradients')
        # parser.add_argument(
        #     '--clip_grad_norm_m', action='store_true', help='whether to clip gradients')
        # parser.add_argument(
        #     '--clip_m', type=float, default=1.0)
        # parser.add_argument(
        #     '--clip_c', type=float, default=5.0)
        # parser.add_argument(
        #     '--weight_decay_c', type=float, default=1e-10, help='adam optimizer weight decay')
        # parser.add_argument(
        #     '--weight_decay_m', type=float, default=1e-10, help='adam optimizer weight decay')
        # parser.add_argument(
        #     '--error_bars', action='store_true', default=False,
        #     help='trains 10 times and return the standard deviation and mean of test loss')
        # # parser.add_argument(
        # #     '--random_search', action='store_true', help='random search over hyperparameters')
        # # parser.add_argument(
        # #     '--continue_from', type=int, default=0, help='continue random search from iteration number')
        #
        # # Dataset options

        # parser.add_argument(
        #     '--marginal_1', default='gamma',
        #     choices=['gaussian', 'uniform', 'gamma', 'lognormal', 'gmm', 'mix_gamma',
        #              'mix_lognormal', 'mix_gauss_gamma'], help='marginal in first dimension')
        # parser.add_argument(
        #     '--marginal_2', default='gamma',
        #     choices=['gaussian', 'uniform', 'gamma', 'lognormal', 'gmm', 'mix_gamma',
        #              'mix_lognormal', 'mix_gauss_gamma'], help='marginal in second dimension')



        #
        # # Options NSF - Copula estimation
        # parser.add_argument(
        #     '--n_layers_c', type=int, default=10, help='Number of spline layers in flow')
        # parser.add_argument(
        #     '--hidden_units_c', type=int, default=16, help='Number of hidden units in spline layer')
        # parser.add_argument(
        #     '--n_blocks_c', type=int, default=3, help='Number of residual blocks in each spline layer')
        # parser.add_argument(
        #     '--tail_bound_c', type=float, default=64, help='Bounds of spline region')
        # parser.add_argument(
        #     '--tails', type=str, default='linear', help='Function type outside spline region')
        # parser.add_argument(
        #     '--n_bins_c', type=int, default=25, help='Number of bins in piecewise spline transform')
        # parser.add_argument(
        #     '--min_bin_height', type=float, default=1e-3, help='Minimum bin height of piecewise transform')
        # parser.add_argument(
        #     '--min_bin_width', type=float, default=1e-3, help='Minimum bin width of piecewise transform')
        # parser.add_argument(
        #     '--min_derivative', type=float, default=1e-3, help='Minimum derivative at bin edges')
        # parser.add_argument(
        #     '--dropout_c', type=float, default=0.15, help='Dropout probability in flow')
        # parser.add_argument(
        #     '--use_batch_norm_c', type=bool, default=True, help='Use batch norm in spline layers')
        # parser.add_argument(
        #     '--unconditional_transform', type=int, default=0, help='Unconditionally transform identity features')
        # parser.add_argument(
        #     '--amsgrad_c', action='store_true', default=False)
        #
        # # NSF Options marginal
        # parser.add_argument(
        #     '--n_layers_m', type=int, default=10, help='Number of spline layers in flow')
        # parser.add_argument(
        #     '--hidden_units_m', type=int, default=128, help='Number of hidden units in spline layer')
        # parser.add_argument(
        #     '--n_blocks_m', type=int, default=2, help='Number of residual blocks in each spline layer')
        # parser.add_argument(
        #     '--n_bins_m', type=int, default=4, help='Number of bins in piecewise spline transform')
        # parser.add_argument(
        #     '--dropout_m', type=float, default=0.15, help='Dropout probability in flow')
        # parser.add_argument(
        #     '--tail_bound_m', type=float, default=8, help='Bounds of spline region')
        # parser.add_argument(
        #     '--identity_init_m', action='store_true')
        # parser.add_argument(
        #     '--tails_m', type=str, default='linear', help='Function type outside spline region')
        #

        self.initialized = True
        return self.parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize()

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()

        self.print_options(opt)
        self.opt = opt
        return self.opt
