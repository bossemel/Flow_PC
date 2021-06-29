set -e

# python3 experiments/exp_marg_flow.py --marginal uniform --exp_name exp_marg_flow_uniform
# python3 experiments/exp_marg_flow.py --marginal gaussian --exp_name exp_marg_flow_gaussian
# python3 experiments/exp_marg_flow.py --marginal gamma --exp_name exp_marg_flow_gamma
# python3 experiments/exp_marg_flow.py --marginal lognormal --exp_name exp_marg_flow_lognormal
python3 experiments/exp_marg_flow.py --marginal gmm --exp_name exp_marg_flow_gmm
python3 experiments/exp_marg_flow.py --marginal mix_gamma --exp_name exp_marg_flow_mix_gamma
python3 experiments/exp_marg_flow.py --marginal mix_lognormal --exp_name exp_marg_flow_mix_lognormal
python3 experiments/exp_marg_flow.py --marginal mix_gauss_gamma --exp_name exp_marg_flow_mix_gauss_gamma
