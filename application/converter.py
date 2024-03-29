import pandas as pd
import os 
import numpy as np
import ray
from filelock import FileLock
from tqdm import tqdm


@ray.remote
def get_concessions_responses(thread_sequence: pd.DataFrame) -> tuple:
    """
    Get the concessions and responses for a thread.
    :param thread_sequence: The thread sequence.
    :return: A tuple of (<concessions>, 
                         <responses>, 
                         <opponent_concessions>, 
                         <opponent_resopnse>).
    """
    # Calculate the concessions 
    price_sequence = thread_sequence['offr_price'].tolist()
    response_sequence = thread_sequence['src_cre_date'].tolist()

    concessions = [np.nan, np.nan]
    responses = [np.nan]

    for ii in range(len(thread_sequence)-1):
        if ii < len(thread_sequence)-2:
            concessions.append(abs(price_sequence[ii+2] - price_sequence[ii]))
        responses.append(response_sequence[ii+1] - response_sequence[ii])

    opponent_concession = [np.nan] + concessions[:-1]
    opponent_response = [np.nan] + responses[:-1]

    return concessions, responses, opponent_concession, opponent_response


@ray.remote
def transform_data(data: pd.DataFrame) -> tuple:
    """
    Generate the concession variable.
    :param data: The data to be transformed.
    :return: A tuple of the transformed data and the number of rows.
    """    
    # Loop through thread ids
    unique_threads = data['anon_thread_id'].unique()

    # Number of unique threads
    n_unique_threads = len(unique_threads)

    # Convert date format 
    data['src_cre_date'] = pd.to_datetime(data['src_cre_date'], format='%d%b%Y %H:%M:%S')

    data_new = None
    counter = 0
    for tt, thread in enumerate(unique_threads):
        if tt % 10000 == 0:
            print('Thread {} out of {}'.format(tt, n_unique_threads))

        # Get index of data for this thread
        thread_index = data.loc[data['anon_thread_id'] == thread].index.tolist()
        
        # Get thread sequence
        thread_sequence = data.loc[thread_index, :]

        # Check if thread sequence is longer than 3
        if len(thread_sequence) >= 4:             
            # Sort by src_cre_date
            thread_sequence = thread_sequence.sort_values(by=['src_cre_date'])
            
            # Check if thread sequence matches desired 0,2,1,.. format
            offer_sequence = thread_sequence['offr_type_id'].tolist()
            correct_sequence = [0] + [2, 1] * int((len(offer_sequence)-1)/2)
            if (offer_sequence == correct_sequence) | (offer_sequence == (correct_sequence + [2])): 
                counter += 1

                # Get concession and response variables
                ray_result = get_concessions_responses.remote(thread_sequence)
                concessions, responses, opp_concessions, opp_resonses = ray.get(ray_result)

                # Add to data
                thread_sequence['concessions'] = concessions
                thread_sequence['response_time'] = responses
                thread_sequence['opp_concessions'] = opp_concessions
                thread_sequence['opp_response_time'] = opp_resonses

                # Convert responses to seconds
                thread_sequence['response_time'] = thread_sequence['response_time'].dt.total_seconds()
                thread_sequence['opp_response_time'] = thread_sequence['opp_response_time'].dt.total_seconds()

                # Extend experience value to all rows
                thread_sequence['slr_hist'] = thread_sequence['slr_hist'].iloc[0]
                thread_sequence['byr_hist'] = thread_sequence['byr_hist'].iloc[0]

                # Add to new dataframe
                if data_new is None: 
                    data_new = thread_sequence
                else:
                    data_new = data_new.append(thread_sequence)

    return data_new, counter
        

def convert_chunk(chunk):
    """
    Converts one chunk and saves it to the csv. 
    :param chunk: The chunk to be converted.
    :return: The number of rows converted.
    """
    # Transform the data
    transformed = transform_data.remote(chunk)
    data, counter = ray.get(transformed)

    # Append to data file
    data.to_csv(processed_file_path, mode='a', header=False, index=False)
    return counter

def convert_chunks(filename):
    """
    Converts the dataset in chunks. 
    :param filename: The name of the file to be converted.
    """
    full_counter = 0
    for cc, chunk in enumerate(pd.read_csv(filename, chunksize=chunksize, usecols=usecols)):
        print('Chunk number {}'.format(cc))

        # Transform the data
        counter = convert_chunk(chunk)

        # Count threads
        full_counter += counter
        print('Counter: {}'.format(full_counter))
    print('Full counter: {}'.format(full_counter))

if __name__ == '__main__':
    # Start Ray.
    ray.init()

    # Load the data
    # exclude: buyer_us, byr_cntry_id, response_time (because faulty), anon_slr_id, anon_byr_id
    usecols = ['anon_thread_id', 
                'offr_type_id', 
                'status_id', 
                'offr_price', 
                'src_cre_date', 
                'slr_hist', 
                'byr_hist', 
                'any_mssg', 
                'fdbk_pstv_src', 
                'fdbk_score_src',
                'anon_slr_id', 
                'anon_byr_id']

    chunksize = 10 ** 6
    filename = os.path.join('datasets', 'ebay_data', 'anon_bo_threads.csv')
    full_counter = 0

    # Create empty header dataset 
    header = pd.DataFrame(columns=['anon_thread_id', 
                                   'anon_byr_id', 
                                   'anon_slr_id', 
                                   'fdbk_score_src',
                                   'fdbk_pstv_src', 
                                   'offr_type_id', 
                                   'status_id', 
                                   'offr_price',
                                   'src_cre_date', 
                                   'slr_hist', 
                                   'byr_hist', 
                                   'any_mssg', 
                                   'concessions',
                                   'response_time', 
                                   'opp_concessions', 
                                   'opp_response_time'])
    # Create empty dataframe at destination
    processed_file_path = os.path.join('datasets', 'ebay_data', 'anon_bo_threads_processed_2.csv')
    header.to_csv(processed_file_path, mode='w', header=True, index=False)

    # Convert the data
    convert_chunks(filename)