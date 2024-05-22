import pandas as pd
import numpy as np
from astroquery.gaia import Gaia
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

file_path = 'C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/gaiaupload.csv'
df = pd.read_csv(file_path)
print(f'Attempting Match for {len(df)} BDBS Stars')

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"

def query_gaia_batch(gaia_ids):
    gaia_ids_str = ', '.join(map(str, gaia_ids))
    query = f"SELECT source_id, teff_gspphot, parallax, parallax_error, pmra, pmra_error, pmdec, pmdec_error FROM gaiadr3.gaia_source WHERE source_id IN ({gaia_ids_str})"
    job = Gaia.launch_job(query)
    result = job.get_results()
    return result

def main(data_frame):
    gaia_ids_list = data_frame['gaia_id'].tolist()
    batch_size = 10000
    progress_bar = tqdm(total=len(gaia_ids_list)//batch_size, desc="Querying Gaia")

    temp_values = []
    parallax_values = []
    parallax_error_values = []
    pmra_values = []
    pmra_error_values = []
    pmdec_values = []
    pmdec_error_values = []
    gaia_ids = []  # from GAIA itself

    with ThreadPoolExecutor() as executor:
        for i in range(0, len(gaia_ids_list), batch_size):
            gaia_ids_batch = gaia_ids_list[i:i + batch_size]
            future = executor.submit(query_gaia_batch, gaia_ids_batch)
            result = future.result()

            for row in result:

                temp_values.append(row['teff_gspphot'])
                parallax_values.append(row['parallax'])
                parallax_error_values.append(row['parallax_error'])
                pmra_values.append(row['pmra'])
                pmra_error_values.append(row['pmra_error'])
                pmdec_values.append(row['pmdec'])
                pmdec_error_values.append(row['pmdec_error'])
                gaia_ids.append(row['source_id'])

            progress_bar.update(1)

    progress_bar.close()

    # Convert lists to DataFrame
    additional_info_df = pd.DataFrame({
        'gaia_id' : gaia_ids,
        'teff_val' : temp_values,
        'parallax': parallax_values,
        'parallax_error': parallax_error_values,
        'pmra': pmra_values,
        'pmra_error': pmra_error_values,
        'pmdec': pmdec_values,
        'pmdec_error': pmdec_error_values,
    })

    # Combine the additional_info_df with the original DataFrame
    df= pd.merge(data_frame, additional_info_df, on='gaia_id', how='left')
    # Save the updated DataFrame
    df['teff_val'] = pd.to_numeric(df['teff_val'], errors='coerce')
    df['parallax'] = pd.to_numeric(df['parallax'], errors='coerce')
    df['parallax_error'] = pd.to_numeric(df['parallax_error'], errors='coerce')
    df['pmra'] = pd.to_numeric(df['pmra'], errors='coerce')
    df['pmra_error'] = pd.to_numeric(df['pmra_error'], errors='coerce')
    df['pmdec'] = pd.to_numeric(df['pmdec'], errors='coerce')
    df['pmdec_error'] = pd.to_numeric(df['pmdec_error'], errors='coerce')
    # No Negative Parallaxes.
    df = df[(df['parallax'] >= 0)]
    df.to_csv('C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/gaiaresults.csv', index=False)

# Run the main function
main(df.copy())

