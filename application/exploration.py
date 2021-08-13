import os
import pandas as pd
import numpy as np 
from eval.plots import histogram
from pathlib import Path


def preprocess(data):
    """
    Preprocess data.
    """
    # Drop duplicate rows
    data = data.drop_duplicates()
    
    # Create counter for thread ids
    counter = data.sort_values(by=['anon_thread_id', 'src_cre_date']).groupby(['anon_thread_id']).cumcount()
    data['offer_counter'] = counter

    # Calculate time since first offer
    counter = data.sort_values(by=['anon_thread_id', 'response_time']).reset_index(drop=True)
    data["time_since_offer"] = data.groupby(['anon_thread_id'])['response_time'].cumsum(axis=0)    

    # Drop na values
    data = data.dropna(subset=['concessions', 'response_time', 'opp_concessions', 'opp_response_time'])

    # Drop threads with offer prices outside 1-1000
    #data = data.loc[data['offr_price'] >= 1]
    #data = data.loc[data['offr_price'] <= 20000]

    # Drop threads with concessions over offer price
    data = data.loc[data['concessions'] <= data['offr_price']]
    data = data.loc[data['opp_concessions'] <= data['offr_price']]

    # Create log transformed variables
    data['log_concessions'] = np.log(data['concessions'] + 1)
    data['log_opp_concessions'] = np.log(data['opp_concessions'] + 1)
    data['log_offr_price'] = np.log(data['offr_price'] + 1)
    data['log_response_time'] = np.log(data['response_time'] + 1)
    data['log_opp_response_time'] = np.log(data['opp_response_time'] + 1)
    data['log_time_since_offer'] = np.log(data['time_since_offer'] + 1)
    data['log_slr_hist'] = np.log(data['slr_hist'] + 1)
    data['log_byr_hist'] = np.log(data['byr_hist'] + 1)

    return data


if __name__ == '__main__':
    pd.options.display.float_format = '{:,.3f}'.format

    chunksize = 10 ** 6
    file_path = os.path.join('datasets', 'ebay_data', 'anon_bo_threads_processed.csv')

    # Read in data
    data = pd.read_csv(file_path, index_col=False)
    
    # Preprocess data
    data = preprocess(data)

    # Save processed data
    file_path_2 = os.path.join('datasets', 'ebay_data', 'anon_bo_threads_processed_2.csv')
    data.to_csv(file_path_2)
    
    # Read in data
    file_path_2 = os.path.join('datasets', 'ebay_data', 'anon_bo_threads_processed_2.csv')
    data = pd.read_csv(file_path_2, index_col=False)
       
    # Describe data
    print(data.filter(items=['offr_price',
                             'offer_counter',
                             'slr_hist', 
                             'byr_hist', 
                             'concessions',
                             'response_time', 
                             'opp_concessions', 
                             'opp_response_time',
                             'time_since_offer']).describe())

    # Create histogram of the offer price
    plots_path = os.path.join('results', 'ebay', 'exploratory')
    Path(plots_path).mkdir(parents=True, exist_ok=True)

    histogram(data['log_concessions'], path=plots_path, var_name='log(concession + 1)', plt_name='concession')
    histogram(data['log_opp_concessions'], path=plots_path, var_name='log(opponent concession + 1)', plt_name='opponent_concession')
    histogram(data['log_offr_price'], path=plots_path, var_name='log(offer price + 1)', plt_name='offer_price')
    histogram(data['log_response_time'], path=plots_path, var_name='log(response time + 1)', plt_name='response_time')
    histogram(data['log_opp_response_time'], path=plots_path, var_name='log(opponent response time + 1)', plt_name='opponent_response_time')
    histogram(data['log_time_since_offer'], path=plots_path, var_name='log(time since offer + 1)', plt_name='time_since_offer')
    histogram(data['offer_counter'], path=plots_path, var_name='offer counter', plt_name='offer_counter')
    histogram(data['log_slr_hist'], path=plots_path, var_name='log(seller history + 1)', plt_name='seller_history')
    histogram(data['log_byr_hist'], path=plots_path, var_name='log(buyer history + 1)', plt_name='buyer_history')

    
    # Create dataset with  
    #   - log(concession + 1)
    #   - log(opponent concession + 1)
    #   - log(offer price + 1)
    #   - log(time since offer + 1)
    #   - offer counter
    #   - log(seller history + 1)
    #   - log(buyer history + 1)
    data_cons = data.filter(items=['log_concessions',
                                  'log_opp_concessions',
                                  'log_offr_price',
                                  'log_time_since_offer',
                                  'offer_counter',
                                  'log_slr_hist',
                                  'log_byr_hist'])

    # Save dataset
    file_path_cons = os.path.join('datasets', 'ebay_data', 'consessions_subset.csv')
    data_cons.to_csv(file_path_cons)




    
