import pandas as pd
from astroquery.simbad import Simbad
from astropy.coordinates import SkyCoord
from astropy import units as u
from concurrent.futures import ThreadPoolExecutor, as_completed

# csv_file_path ='C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/Disk_BC.csv'
# csv_file_path = 'C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/gaiaupload.csv'
csv_file_path = 'C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/BDBS_W_Parallax.csv'
df = pd.read_csv(csv_file_path)
print(f"Loaded {len(df)} stars from {csv_file_path}.")

# Initialize Simbad query
simbad_query = Simbad()
simbad_query.add_votable_fields('otype', 'flux(J)', 'flux(H)', 'flux(K)')

def query_simbad(row):
    ra = row['ra']
    dec = row['dec']
    try:
        result_table = simbad_query.query_region(
            SkyCoord(ra, dec, unit=(u.deg, u.deg), frame='icrs'),
            radius='0d0m00.5s'
        )
        if result_table is not None and len(result_table) > 0:
            result_df = result_table.to_pandas()
            return {
                'ra': ra, 'dec': dec,
                'star_name': result_df['MAIN_ID'].values[0],
                'object_type': result_df['OTYPE'].values[0],
                'ra_prec': result_df['RA_PREC'].values[0],
                'dec_prec': result_df['DEC_PREC'].values[0],
                'jmag': result_df['FLUX_J'].values[0],
                'hmag': result_df['FLUX_H'].values[0],
                'kmag': result_df['FLUX_K'].values[0],
                **row[['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']].to_dict()
            }
    except Exception as e:
        print(f"Error querying Simbad for RA: {ra}, Dec: {dec}: {e}")
    return None

# Number of threads to use
num_threads = 6

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(query_simbad, row) for _, row in df.iterrows()]
    results = [future.result() for future in as_completed(futures) if future.result() is not None]

if results:
    matched_stars_df = pd.DataFrame(results)
    matched_stars_df.to_csv('allbdbssimbad.csv', index=False)
    print("Matched stars saved to 'simbad_matched.csv'.")
else:
    print("No Simbad results found for the given stars.")
