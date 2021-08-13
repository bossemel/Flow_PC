import os
import pandas as pd
import numpy as np 
from eval.plots import histogram
from pathlib import Path


if __name__ == '__main__':
 
    # Load dataset
    file_path_cons = os.path.join('datasets', 'ebay_data', 'consessions_subset.csv')
    data = pd.read_csv(file_path_cons)
    
    print(data.head())