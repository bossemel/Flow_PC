import os
import pandas as pd
import numpy as np 


def transform_data(data):
    """
    Generate the concession variable
    """    
    # Loop through thread ids
    unique_threads = data['anon_thread_id'].unique()

    # Number of unique threads
    n_unique_threads = len(unique_threads)

    # for tt, thread in enumerate(unique_threads):
    #     print('Thread {} out of {}'.format(tt, n_unique_threads))

    #     # Get index of data for this thread
    #     thread_index = data.loc[data['anon_thread_id'] == thread].index.tolist()
        
    #     # Get thread sequence
    #     thread_sequence = data.loc[thread_index, :]


if __name__ == '__main__':
    pd.options.display.float_format = '{:,.3f}'.format

    chunksize = 10 ** 6
    file_path = os.path.join('datasets', 'ebay_data', 'anon_bo_threads_processed.csv')

    # Read in data
    data = pd.read_csv(file_path, index_col=False)
    
    # Drop duplicate rows
    data = data.drop_duplicates()
    
    # Drop na values
    data = data.dropna(subset=['concessions', 'response_time', 'opp_concessions', 'opp_response_time'])

    # Drop threads with offer prices outside 1-1000
    data = data.loc[data['offr_price'] >= 1]
    data = data.loc[data['offr_price'] <= 1000]

    # Drop threads with concessions over offer price
    data = data.loc[data['concessions'] <= data['offr_price']]
    data = data.loc[data['opp_concessions'] <= data['offr_price']]

    # Describe data
    print(data.filter(items=['fdbk_score_src', 
                             'fdbk_pstv_src']).describe())
