
import pandas as pd
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
from concurrent.futures import ThreadPoolExecutor, as_completed

csv_file_path = 'C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/Pan-Starrs_Data/PS_filtered.csv'
df = pd.read_csv(csv_file_path)
df = df[['raMean', 'decMean', 'objID']]
print(f"Loaded {len(df)} stars from {csv_file_path}.")

# Initialize Simbad query
simbad_query = Simbad()
simbad_query.add_votable_fields('otype', 'flux(J)', 'flux(H)', 'flux(K)', 'ra(d)', 'dec(d)') # Added RA and Dec in degrees

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
            # Retrieve RA and Dec in degrees from SIMBAD
            simbad_ra = result_df['RA_d'].values[0]
            simbad_dec = result_df['DEC_d'].values[0]
            return {
                'PS_id': id_, 
                'ra': ra, 'dec': dec,
                'ra_err': ra_err, 'dec_err': dec_err, # Include Pan-STARRS errors
                'simbad_ra': simbad_ra, 'simbad_dec': simbad_dec, # Include SIMBAD coordinates
                'star_name': result_df['MAIN_ID'].values[0],
                'object_type': result_df['OTYPE'].values[0],
                'ra_prec': result_df['RA_PREC'].values[0],
                'dec_prec': result_df['DEC_PREC'].values[0],
                'j': result_df['FLUX_J'].values[0],
                'h': result_df['FLUX_H'].values[0],
                'k': result_df['FLUX_K'].values[0],
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
    matched_stars_df.to_csv('simbad_matched.csv', index=False)
    print("Matched stars with distances saved to 'simbad_matched.csv'.")
else:
    print("No Simbad results found for the given stars.")

