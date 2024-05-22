import pandas as pd
import numpy as np
import gzip

def process_data(file_path):
    processed_data = []
    with gzip.open(file_path, 'rt') as file:
        for chunk in pd.read_csv(file, chunksize=5000000):
            print("Number of Stars Before Selections:", len(chunk))
            
            selected_stars = chunk.replace(99.999, np.nan).dropna()

            error_threshold = 0.1
            
            for mag_band in ['u', 'g', 'r', 'i', 'z', 'y']:
                err_column = f'{mag_band}err'
                err_column2 = f'{mag_band}_err2'
                n_meas = f'{mag_band}_number_measurements'
                err_flag = f'{mag_band}_error_flag'
                sky_flag = f'{mag_band}_sky_flag'
                shape_flag = f'{mag_band}_shape_flag'
                overlap_flag = f'{mag_band}_overlap_flag'

                selected_stars = selected_stars[
                                (selected_stars[err_column] <= error_threshold)
                                & (selected_stars[err_column] <= selected_stars[err_column2])
                                & (selected_stars[n_meas] >= 5) 
                                & (selected_stars[err_flag] <= 1)     
                                & (selected_stars[sky_flag] <= 1)      
                                & (selected_stars[shape_flag] <= 1)
                                & (selected_stars[overlap_flag] <= 1)    
                                ]

            print("Number of Stars After Selections:", len(selected_stars))
            print("Number of Stars Cut:", len(chunk) - len(selected_stars))

            columns_keep=['ra', 'dec', 'umag', 'gmag','rmag', 'imag', 'zmag', 'ymag', 'gaia_id', 'BDBS_ID']

            df_nn = selected_stars[columns_keep].copy()

            processed_data.append(df_nn)

    processed_data = pd.concat(processed_data, ignore_index=True)
    print(len(processed_data))
    processed_file_path = 'newprocessed_data.parquet'

    processed_data.to_parquet(processed_file_path, index=False, compression='snappy')

    print("Processed data saved to:", processed_file_path)

if __name__ == "__main__":
    input_file_path = '/N/project/catypGC/BDBS/stellaridentification/bdbs_output.csv.gz'
    process_data(input_file_path)
