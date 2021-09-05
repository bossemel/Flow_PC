# Flow-PC

A deep-learning based PC algorithm for causal discovery. The conditional independence test is based on conditional mutual information which is in turn estimated by conditional copula density. The conditional copula density is measured using Normalizing Flows for both the marginal and the copula. 

### Installation

For installation, first create a virtual environment and install the required packages:

```
python3 -m venv .env
source .env/bin/activate
pip3 install -r requirements.txt
pip3 install -e .
```

## Experiments

To run the experiments on the copula flow, the marginal flow and PC algorithm with toy data, run ```experiments/exp_cop_flow.py```, ```experiments/exp_marg_flow.py``` and ```experiments/exp_pc.py```. The results will be saved in ```results/```.

## Application

Download the [eBay negotation data at a thread level from NBER](https://data.nber.org/data-appendix/w24306/bargaining/anon_bo_threads.csv.gz) and save it under ```datasets/ebay_data/```. Then, run ```application/converter.py```. Then create the summary statistics and histogram by running ```applications/exploration.py```. To run the example application, run ```applications/estimation.py```. 
