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

To run the experiments on the copula flow, the marginal flow and PC algorithm with toy data, run:

```
bash scripts/simulation_experiments.sh
```

## Application

Download the [eBay negotation data at a thread level from NBER](https://data.nber.org/data-appendix/w24306/bargaining/anon_bo_threads.csv.gz) and save it under ```datasets/ebay_data/```. Then, run:

```
bash scripts/application_experiments.sh
```