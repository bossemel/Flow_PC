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

    for tt, thread in enumerate(unique_threads):
        print('Thread {} out of {}'.format(tt, n_unique_threads))

        # Get index of data for this thread
        thread_index = data.loc[data['anon_thread_id'] == thread].index.tolist()
        
        # Get thread sequence
        thread_sequence = data.loc[thread_index, :]

        if len(thread_sequence) > 4:
            # Sort by src_cre_date
            thread_sequence = thread_sequence.sort_values(by=['src_cre_date'])
            concession_sequence = thread_sequence['concession'].tolist()
            response_sequence = thread_sequence['response_time'].tolist()

            opponent_concession = [np.nan, np.nan, np.nan]
            opponent_response = [np.nan, np.nan]
            for ii in range(2, len(concession_sequence)):
                if ii >= 3:
                    opponent_concession.append(concession_sequence[ii-1])
                opponent_response.append(response_sequence[ii-1])

            thread_sequence['opp_concessions'] = opponent_concession
            thread_sequence['opp_response_time'] = opponent_response

            exit()

if __name__ == '__main__':
    chunksize = 10 ** 6
    file_path = os.path.join('datasets', 'ebay_data', 'anon_bo_threads_processed.csv')

    for cc, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize, index_col=False, dtype={'src_cre_date': 'str', 'response_time': 'str'}, parse_dates=['src_cre_date', 'response_time', 'opp_response_time'])):
        print('Chunk number {}'.format(cc))
        transform_data(chunk)