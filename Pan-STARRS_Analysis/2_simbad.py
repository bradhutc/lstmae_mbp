import pandas as pd
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
DATA_DIR = "/N/project/catypGC/Bradley/Data_ML"
PLOT_DIR = "/N/project/catypGC/Bradley/Plots_ML"

csv_file_path = os.path.join(DATA_DIR, 'PS_Clean.csv')
output_file_path = os.path.join(DATA_DIR, 'simbad_matched.csv')

try:
    df = pd.read_csv(csv_file_path)
    df = df[['raMean', 'decMean', 'objID', 'raMeanErr', 'decMeanErr']]
    print(f"Loaded {len(df)} stars from {csv_file_path}.")
except FileNotFoundError:
    print(f"Error: The file {csv_file_path} was not found.")
    exit(1)
except pd.errors.EmptyDataError:
    print(f"Error: The file {csv_file_path} is empty.")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading the file: {e}")
    exit(1)

# Initialize Simbad query
simbad_query = Simbad()
simbad_query.add_votable_fields('otype', 'ra(d)', 'dec(d)')

def query_simbad(row):
    ra = row['raMean']
    ra_err = row['raMeanErr']
    dec = row['decMean']
    dec_err = row['decMeanErr']
    id_ = row['objID']
    
    try:
        result_table = simbad_query.query_region(
            SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'),
            radius='0d0m01s'
        )
        if result_table is not None and len(result_table) > 0:
            result_df = result_table.to_pandas()
            simbad_ra = result_df['RA_d'].values[0]
            simbad_dec = result_df['DEC_d'].values[0]
            return {
                'PS_id': id_, 
                'ra': ra, 'dec': dec,
                'ra_err': ra_err, 'dec_err': dec_err,
                'simbad_ra': simbad_ra, 'simbad_dec': simbad_dec,
                'star_name': result_df['MAIN_ID'].values[0],
                'object_type': result_df['OTYPE'].values[0],
                'ra_prec': result_df['RA_PREC'].values[0],
                'dec_prec': result_df['DEC_PREC'].values[0],
            }
    except Exception as e:
        print(f"Error querying Simbad for RA: {ra}, Dec: {dec}: {e}")
    return None

num_threads = 5
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(query_simbad, row) for _, row in df.iterrows()]
    results = [future.result() for future in as_completed(futures) if future.result() is not None]

if results:
    matched_stars_df = pd.DataFrame(results)
    try:
        matched_stars_df.to_csv(output_file_path, index=False)
        print(f"Matched stars with distances saved to '{output_file_path}'.")
    except Exception as e:
        print(f"Error saving the matched stars: {e}")
else:
    print("No Simbad results found for the given stars.")