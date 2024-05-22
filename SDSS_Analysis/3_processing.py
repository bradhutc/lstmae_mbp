import pandas as pd
from astropy.coordinates import SkyCoord
import astropy.units as u
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from astropy.units import radian
from scipy import stats

def load_data(sdss_file, matched_file):
    """
    Load the SDSS and matched data from CSV files.
    
    Parameters:
        sdss_file (str): Path to the SDSS data file.
        matched_file (str): Path to the matched data file.
    
    Returns:
        tuple: A tuple containing two pandas DataFrames: SDSS data and matched data.
    """
    sdss_data = pd.read_csv(sdss_file)
    matched_data = pd.read_csv(matched_file)
    return sdss_data, matched_data

def compute_angular_separation(matched_data):
    """
    Compute the angular separation between SDSS and SIMBAD coordinates in the matched data,
    using the custom angular_separation function.
    
    Parameters:
        matched_data (pd.DataFrame): DataFrame containing the matched data with 'ra', 'dec', 'simbad_ra', 'simbad_dec'.
    
    Returns:
        pd.DataFrame: The input DataFrame with an added column for angular separation in arcseconds.
    """
    matched_data['ra'] = pd.to_numeric(matched_data['ra'], errors='coerce')
    matched_data['dec'] = pd.to_numeric(matched_data['dec'], errors='coerce')
    
    # Convert 'simbad_ra' and 'simbad_dec' to numeric types
    matched_data['simbad_ra'] = pd.to_numeric(matched_data['simbad_ra'], errors='coerce')
    matched_data['simbad_dec'] = pd.to_numeric(matched_data['simbad_dec'], errors='coerce')
    
    # Extract coordinates
    ra_sdss = matched_data['ra'].values
    dec_sdss = matched_data['dec'].values
    ra_simbad = matched_data['simbad_ra'].values
    dec_simbad = matched_data['simbad_dec'].values
    
    # Create SkyCoord objects
    coord_sdss = SkyCoord(ra=ra_sdss*u.deg, dec=dec_sdss*u.deg)
    coord_simbad = SkyCoord(ra=ra_simbad*u.deg, dec=dec_simbad*u.deg)
    
    # Compute separation and add it to the DataFrame
    separation = coord_sdss.separation(coord_simbad).arcsec
    matched_data['angular_separation'] = separation
    
    plt.figure(figsize=(10, 6))
    plt.hist(matched_data['angular_separation'], bins=100, color='black')
    plt.title('Angular Separation between SDSS and SIMBAD Coordinates')
    plt.xlabel('Angular Separation (arcseconds)')
    plt.ylabel('Count')
    plt.xlim(0, 1000)
    plt.tight_layout()
    plt.savefig('angular_separation.png', format='png', dpi=300)
    
    return matched_data

def filter_stars(sdss_data, matched_data):
    """
    Filter SDSS data to retain only star types based on matched data.
    
    Parameters:
        sdss_data (pd.DataFrame): DataFrame containing the SDSS data.
        matched_data (pd.DataFrame): DataFrame containing the matched data with stellar type identification.
    
    Returns:
        pd.DataFrame: SDSS data filtered to include only stars.
    """
    # Define a list of star labels
    star_labels = [
        'Star', 'RRLyrae', 'RGB*', 'PulsV*', 'BlueStraggler', 'HighPM*',
        'ChemPec*', 'EmLine*', 'WhiteDwarf', 'HorBranch*', 'HighVel*', 
        'Variable*', 'Low-Mass*', 'HorBranch*_Candidate', 'EclBin', 'SB*', 
        'C*_Candidate', 'HotSubdwarf', '**', 'blue', 'HotSubdwarf_Candidate', 
        'Optical', 'WhiteDwarf_Candidate', 'C*', 'LongPeriodV*', 
        'ChemPec*_Candidate', 'X', 'RSCVnV*', 'SXPheV*', 
        'YSO_Candidate', 'EclBin_Candidate', 'UV', 'delSctV*', 'Eruptive*', 
        'AGB*', 'Mira', 'Cepheid', 'Type2Cep',
        'alf2CVnV*', 'EllipVar', 'ClassicalCep'
    ]
    # also filter out the variable stars
    
    
# List of variable or pulsating star types

    # variable_stars = [

    #     'RRLyrae', 'PulsV*', 'Variable*', 'EclBin', 'LongPeriodV*', 'RSCVnV*', 

    #     'SXPheV*', 'delSctV*', 'Eruptive*', 'Mira', 'Cepheid', 'Type2Cep',

    #     'alf2CVnV*', 'EllipVar', 'ClassicalCep'

    # ]



    # # Filter the list to exclude the variable or pulsating stars

    # star_labels = [star for star in star_labels if star not in variable_stars]
        
    
    
    non_starlike_labels = matched_data[~matched_data['object_type'].isin(star_labels)]
    stars = matched_data[matched_data['object_type'].isin(star_labels)]
    # Identify SDSS entries to remove by matching 'ra' and 'dec' with non-star-like entries
    # Assuming 'ra' and 'dec' can uniquely identify each entry
    sdss_to_remove = sdss_data.merge(non_starlike_labels[['ra', 'dec']], on=['ra', 'dec'])
    
    # Remove these entries from sdss_data
    sdss_data_filtered = sdss_data[~sdss_data.index.isin(sdss_to_remove.index)]
    
    print(f'Removed {len(sdss_data) - len(sdss_data_filtered)} non-star-like entries from SDSS.')
    
    matched_data_filtered = pd.merge(stars, sdss_data[['ra', 'dec', 'u', 'g', 'r', 'i', 'z']],
                    on=['ra', 'dec'], 
                    how='left')
    print(f'Retained {len(matched_data_filtered)} star entries in matched data.')
    return sdss_data_filtered, matched_data_filtered

def plot_object_type_distribution(matched_data):
    """
    Plot the distribution of object types in the matched data.
    
    Parameters:
        matched_data (pd.DataFrame): DataFrame containing the matched data with object types.
    """
    
    # omit plotting 'Star' object type
    all = matched_data
    matched_data = matched_data[matched_data['object_type'] != 'Star']
    print(f'Matched stars with object type star: {len(matched_data)-len(all)}')
    # only include the top 15 object types
    matched_data = matched_data[matched_data['object_type'].isin(matched_data['object_type'].value_counts().index[:12])]
    plt.figure(figsize=(12, 6))
    sns.countplot(data=matched_data, x='object_type', order = matched_data['object_type'].value_counts().index, color='black')
    plt.title('Distribution of Object Types from SIMBAD')
    plt.xlabel('Object Type')
    plt.ylabel('Count')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('object_type_distribution.png', format='png', dpi=300)

def plot_galactic_positions(data):
    """
    Plot the galactic positions of objects in the matched data.
    
    Parameters:
        matched_data (pd.DataFrame): DataFrame containing the matched data with l and b.
    """
    print(data.head())
    plt.figure(figsize=(10, 6))
    plt.scatter(data['l'],data['b'], s=1, alpha=0.5)
    plt.title('Galactic Positions of Objects')
    plt.xlabel('Galactic Longitude (l)')
    plt.ylabel('Galactic Latitude (b)')
    plt.tight_layout()
    plt.show()

def plot_color_magnitude(data):
    """
    Plot the color-magnitude diagram of objects in the matched data.
    
    Parameters:
        matched_data (pd.DataFrame): DataFrame containing the matched data with 'g', 'r', and 'i' magnitudes.
    """
    data['g-i'] = data['g'] - data['i']
    data['g-r'] = data['g'] - data['r']

    plt.figure(figsize=(10, 6))
    # Plot the general population of stars
    plt.hexbin(data['g-i'], data['i'], gridsize=300, cmap='viridis', mincnt=1, alpha=0.5)
    plt.gca().invert_yaxis()
    plt.title('Color-Magnitude Diagram for SDSS Stars')
    plt.xlabel('g-i')
    plt.ylabel('i')
    # plt.legend(title='Object Type')
    plt.tight_layout()
    plt.savefig('fully_cleaned_gi.png')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    plt.hexbin(data['g-r'], data['r'], gridsize=300, cmap='viridis', mincnt=1, alpha=0.5)
    plt.gca().invert_yaxis()
    plt.title('Color-Magnitude Diagram for SDSS Stars')
    plt.xlabel('g-r')
    plt.ylabel('r')
    # plt.legend(title='Object Type')
    plt.tight_layout()
    plt.savefig('fully_cleaned_gr.png')
    plt.show()
    
    
def is_anomalous(magnitudes, threshold):
    """
    Checks if the given sequence of magnitudes exhibits non-monotonic behavior with changes
    exceeding a specified threshold, indicating an anomaly in the SED.
    
    Parameters:
        magnitudes (list): List of magnitudes in sequential bands.
        threshold (float): The change threshold to qualify as an anomaly.
        
    Returns:
        bool: True if an anomaly is detected, False otherwise.
    """
    for i in range(1, len(magnitudes) - 1):
        if ((magnitudes[i-1] - magnitudes[i] > threshold and magnitudes[i+1] - magnitudes[i] > threshold) or \
            (magnitudes[i] - magnitudes[i-1] > threshold and magnitudes[i] - magnitudes[i+1] > threshold)):
            return True  # Anomaly detected due to significant fluctuation
    return False
def plot_anomalies_vs_threshold(data, start=0.1, stop=2.0, step=0.1):
    """
    Plots the number of anomalous stars detected as a function of varying threshold values.

    Parameters:
        data (pd.DataFrame): DataFrame containing the star data with 'u', 'g', 'r', 'i', 'z' magnitudes.
        start (float): Starting value for the threshold range.
        stop (float): Stopping value for the threshold range.
        step (float): Step size for the threshold range.
    """
    thresholds = np.arange(start, stop + step, step)
    anomaly_counts = []

    for threshold in thresholds:
        data['anomalous'] = data.apply(lambda row: is_anomalous([row['u'], row['g'], row['r'], row['i'], row['z']], threshold=threshold), axis=1)
        anomaly_count = data['anomalous'].sum()
        anomaly_counts.append(anomaly_count)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, anomaly_counts, marker='o', linestyle='--', color='black')
    plt.title('Anomalous Stars Detected vs. Magnitude Change Threshold')
    plt.xlabel('Magnitude Change Threshold')
    plt.ylabel('Number of Anomalous Stars Detected')
    plt.grid(True)
    plt.savefig('anomalies_vs_threshold.png', format='png', dpi=300)
    plt.show()
def plot_sample_stars(data, sample_size=10, title='SDSS Star Magnitudes'):
    """
    Plot magnitudes for a sample of stars across the 'u', 'g', 'r', 'i', 'z' bands.

    Parameters:
        data (pd.DataFrame): DataFrame containing the star data.
        sample_size (int): Number of stars to sample and plot.
        title (str): Title for the plot.
    """
    sample_stars = data.sample(n=sample_size)
    plt.figure(figsize=(10, 6))
    for _, star in sample_stars.iterrows():
        plt.plot(['u', 'g', 'r', 'i', 'z'], [star[band] for band in ['u', 'g', 'r', 'i', 'z']], marker='o', alpha=0.7)
    plt.gca().invert_yaxis()
    plt.xlabel('Band')
    plt.ylabel('Magnitude')
    plt.title(title)
    if title == 'Anomalous Data: SDSS Star Magnitudes':
        plt.savefig('anomalous_stars.png', format='png', dpi=300)
    else:
        plt.savefig('cleaned_stars.png', format='png', dpi=300)
    plt.show()
if __name__ == "__main__":
    sdss_data, matched_data = load_data('Stars_cleaned.csv', 'simbad_matched.csv')
    # sdss_data['g-i'] = sdss_data['g'] - sdss_data['i']
    # sdss_data['g-r'] = sdss_data['g'] - sdss_data['r']
    
    # plt.figure(figsize=(10, 6))
    # plt.hexbin(sdss_data['g-i'], sdss_data['i'], gridsize=300, cmap='viridis', mincnt=1, alpha=0.5)
    # plt.gca().invert_yaxis()
    # plt.xlabel('g-i')
    # plt.ylabel('i')
    # plt.title('Color-Magnitude Diagram for SDSS Stars')
    # plt.tight_layout()
    # plt.savefig('preclean_gi.png')
    
    # plt.figure(figsize=(10, 6))
    # plt.hexbin(sdss_data['g-r'], sdss_data['r'], gridsize=300, cmap='viridis', mincnt=1, alpha=0.5)
    # plt.gca().invert_yaxis()
    # plt.xlabel('g-r')
    # plt.ylabel('r')
    # plt.title('Color-Magnitude Diagram for SDSS Stars')
    # plt.tight_layout()
    # plt.savefig('preclean_gr.png')
    
    
    
    print(f"Initial SDSS stars: {len(sdss_data)}")

    # Filter based on object type
    sdss_data_filtered, matched_data_filtered = filter_stars(sdss_data, matched_data)
    print(f"After filtering non-star-like: {len(sdss_data_filtered)}")
    # plot_anomalies_vs_threshold(sdss_data_filtered)
    # plot_object_type_distribution(matched_data_filtered)
    # Apply magnitude fluctuation filter
    threshold = 0.25
    sdss_data_filtered['anomalous_fluctuation'] = sdss_data_filtered.apply(
        lambda row: is_anomalous([row['u'], row['g'], row['r'], row['i'], row['z']], threshold=threshold),
        axis=1
    )
    cleaned_data = sdss_data_filtered[~sdss_data_filtered['anomalous_fluctuation']]
    print(f"After removing magnitude fluctuations: {len(cleaned_data)}")

    matched_data_filtered.to_csv('simbad_matched_cleaned.csv', index=False)
    # Save the cleaned dataset
    cleaned_data.to_csv('cleaned_stellar_data.csv', index=False)

    anomalies = sdss_data_filtered[sdss_data_filtered['anomalous_fluctuation']]

    # Plot for cleaned data
    plot_sample_stars(cleaned_data, sample_size=10, title='Cleaned Data: SDSS Star Magnitudes')

    # Plot for anomalous data
    plot_sample_stars(anomalies, sample_size=10, title='Anomalous Data: SDSS Star Magnitudes')
    
    plot_color_magnitude(cleaned_data)
    
    int_data = cleaned_data[(cleaned_data['g-r'] >2)]
    # plot_sample_stars(int_data, sample_size=10, title='Cleaned Data: SDSS Star Magnitudes')