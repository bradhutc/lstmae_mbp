import pandas as pd
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy

color_indices = ['u-g', 'u-r', 'u-i', 'u-z', 'u-y', 'g-r', 'g-i', 'g-z', 'g-y', 'r-i', 'r-z', 'r-y', 'i-z', 'i-y', 'z-y']
absolute_magnitudes = ['U','G','R','I','Z','Y']
apparent_magnitudes = ['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']

def ProcessGaia(df):
    for index in color_indices:
        color_band_1, color_band_2 = index.split('-')
        df[f'{index}'] = df[f'{color_band_1}mag'] - df[f'{color_band_2}mag']

    print(len(df))
    
    df = df[(df['parallax'] > 0.0) | (df['parallax_error'] <= 0.05)]

    df['d_kpc'] = 1 / (df['parallax'])  # Convert parallax from milliarcseconds to arcseconds

    for i in range(len(absolute_magnitudes)):
        df[f'{absolute_magnitudes[i]}'] = df[f'{apparent_magnitudes[i]}'] - 5 * np.log10((1000/df['parallax'])/10)

    coords = SkyCoord(ra=df['ra'].values*u.degree, dec=df['dec'].values*u.degree, frame='icrs')
    gal = coords.transform_to('galactic')
    t = astropy.table.QTable([gal.l, gal.b], names=('l', 'b'))
    lb_df = t.to_pandas()

    df['l'] = lb_df['l']
    df['b'] = lb_df['b']
    
    for i in range(len(df)):
        if df['l'].values[i] > 5.0:
            df['l'].values[i] -= 360

    for column in df.columns:
        if column not in ['bdbs_id', 'gaia_id']:
            df[column] = df[column].round(5)


    processed_file_path = 'Cleaned_Gaia_Results.csv'
    df.to_csv(processed_file_path, index=False)
    print("Processed data saved to:", processed_file_path)

def ProcessBDBS(df):
    for index in color_indices:
        color_band_1, color_band_2 = index.split('-')
        df[f'{index}'] = df[f'{color_band_1}mag'] - df[f'{color_band_2}mag']

    print(len(df))

    coords = SkyCoord(ra=df['ra'].values*u.degree, dec=df['dec'].values*u.degree, frame='icrs')
    gal = coords.transform_to('galactic')
    t = astropy.table.QTable([gal.l, gal.b], names=('l', 'b'))
    lb_df = t.to_pandas()

    df['l'] = lb_df['l']
    df['b'] = lb_df['b']
    
    for i in range(len(df)):
        if df['l'].values[i] > 5.0:
            df['l'].values[i] -= 360

    for column in df.columns:
        if column not in ['bdbs_id', 'gaia_id']:
            df[column] = df[column].round(5)

    processed_file_path = 'Cleaned_BDBS_Results.csv'
    df.to_csv(processed_file_path, index=False)
    print("Processed data saved to:", processed_file_path)

if __name__ == '__main__':
    File_Path = ['C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/gaiaresults.csv', 'C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/gaiaupload.csv']
    df_gaia = pd.read_csv(File_Path[0]).drop(columns=['teff_val'])
    ProcessGaia(df_gaia)
    df_bdbs = pd.read_csv(File_Path[1])
    ProcessBDBS(df_bdbs)