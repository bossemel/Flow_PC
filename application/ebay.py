import pandas as pd
import os 
import numpy as np


def transform_data(data):
    """
    Generate the concession variable
    """
    # Loop through thread ids
    unique_threads = data['anon_thread_id'].unique()
    data['src_cre_date'] = pd.to_datetime(data['src_cre_date'], format='%d%b%Y %H:%M:%S')
    counter = 0
    for thread in unique_threads:
        if len(data.loc[data['anon_thread_id'] == thread]) >= 3: 
            
            thread_sequence = data[data['anon_thread_id'] == thread]
            
            # Sort by src_cre_date
            thread_sequence = thread_sequence.sort_values(by=['src_cre_date'])
            offer_sequence = thread_sequence['offr_type_id'].tolist()
            correct_sequence = [0] + [2, 1] * int((len(offer_sequence)-1)/2)
            if (offer_sequence == correct_sequence) | (offer_sequence == (correct_sequence + [2])): # Todo: recheck if this is getting all of them
                counter += 1
                price_sequence = thread_sequence['offr_price'].tolist()
                concession = [np.nan, np.nan]
                for ii in range(len(price_sequence)-2):
                    concession.append(abs(price_sequence[ii+2] - price_sequence[ii]))
                data.loc[data['anon_thread_id'] == thread, 'concession'] = concession
                response_sequence = thread_sequence['src_cre_date'].tolist()
                responses = [np.nan]
                for ii in range(len(response_sequence)-1):
                    responses.append(response_sequence[ii+1] - response_sequence[ii])
                data.loc[data['anon_thread_id'] == thread, 'used_response_time'] = responses
            else:
                data = data[data['anon_thread_id'] != thread]
        else: 
            data = data[data['anon_thread_id'] != thread]
    
    return data
        

    
    #data_grouped = data.groupby('anon_thread_id')
    #print(data_grouped) 


if __name__ == '__main__':
    
    # Load the data
    data = pd.read_csv(os.path.join('datasets', 'ebay_data', 'anon_bo_threads.csv'), nrows=10000)
    #print(data)

    # Transform the data 
    data = transform_data(data)
    print(data.head())
    # data = transform_ebay(data)

