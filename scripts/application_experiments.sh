set -3

# Note: Please make sure that the NBER dataset is daved under `datasets/ebay_data/.

# Converting the dataset
python3 application/converter.py

# Filtering and exploring the dataset
python3 applications/exploration.py

# Estimating three causal graphs
python3 applications/estimation.py

# Renaming nodes and plotting the graphs
python3 applications/graph_viz.py