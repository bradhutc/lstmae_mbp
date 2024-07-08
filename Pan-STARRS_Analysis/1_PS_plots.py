import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import seaborn as sns 
import astropy.units as u 
from astropy.coordinates import SkyCoord
import sfdmap
from matplotlib.ticker import AutoMinorLocator

def set_plot_style():
    plt.style.use('default')
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['axes.spines.top'] = False
    plt.rcParams['axes.spines.right'] = False
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['legend.fontsize'] = 12

def customize_axes(ax):
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in')
    ax.tick_params(which='major', length=10)
    ax.tick_params(which='minor', length=5)
def plot_mag_vs_err(df, output_path):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df['gMeanPSFMag'], df['gMeanPSFMagErr'], s=1, alpha=0.5, color='black', label='g-band')
    ax.scatter(df['rMeanPSFMag'], df['rMeanPSFMagErr'], s=1, alpha=0.5, color='red', label='r-band')
    ax.scatter(df['iMeanPSFMag'], df['iMeanPSFMagErr'], s=1, alpha=0.5, color='blue', label='i-band')
    ax.scatter(df['zMeanPSFMag'], df['zMeanPSFMagErr'], s=1, alpha=0.5, color='green', label='z-band')
    ax.scatter(df['yMeanPSFMag'], df['yMeanPSFMagErr'], s=1, alpha=0.5, color='orange', label='y-band')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Magnitude Error')
    ax.set_title('Magnitude vs. Magnitude Error')
    ax.legend()
    customize_axes(ax)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


def plot_b_vs_l(df):
    set_plot_style()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='aitoff')
    
    l_rad = np.deg2rad(df['l'] - 180)  # Shift l by 180 degrees
    b_rad = np.deg2rad(df['b'])

    ax.scatter(l_rad, b_rad, s=1, alpha=0.5, color='black', label='Filtered Pan-STARRS Stars')
    customize_axes(ax)
    ax.set_xlabel('Galactic Longitude (l)')
    ax.set_ylabel('Galactic Latitude (b)')

    
    # ax.grid(True)
    ax.legend()
    plt.savefig('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/galactic_projection.png')
    # plt.show()



def plot_individual_histograms(df, output_dir):
    columns_to_plot = df.columns.drop('objID')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for column in columns_to_plot:
        set_plot_style()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(df[column], bins=50, edgecolor='black', density=False, alpha=0.7, color='maroon')
        ax.set_title(column)
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
        # Calculate and display statistics in the legend
        mean = df[column].mean()
        median = df[column].median()
        std = df[column].std()
        min_val = df[column].min()
        max_val = df[column].max()
        
        legend_text = f"Mean: {mean:.2f}\nMedian: {median:.2f}\nStd: {std:.2f}\nMin: {min_val:.2f}\nMax: {max_val:.2f}"
        ax.legend([legend_text], loc='upper right')
        customize_axes(ax)
        filename = f"{column}_histogram.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300)
        plt.close(fig)
    print(f"Histograms saved in the directory: {output_dir}")
def plot_band_measurements(df):
    band_columns = ['gMeanPSFMagNpt', 'rMeanPSFMagNpt', 'iMeanPSFMagNpt', 'zMeanPSFMagNpt', 'yMeanPSFMagNpt']
    band_labels = ['g', 'r', 'i', 'z', 'y']
    colors = ['green', 'red', 'navy', 'maroon', 'black']
    
    measurements = []
    for column in band_columns:
        measurements.append(df[column].sum())
    
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(band_labels, measurements, color=colors, alpha=0.7)
    ax.set_xlabel('Photometric Band')
    ax.set_ylabel('Number of Measurements')
    customize_axes(ax)
    
    # Add labels on top of each bar
    for i, v in enumerate(measurements):
        ax.text(i, v + 0.1, str(int(v)), ha='center')
    
    plt.tight_layout()
    plt.savefig('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/band_measurements.png', dpi=300)

def plot_b_vs_l_zoomed(df, l_center, b_center, name, radius=5):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))
   
    # Calculate the angular separation between each point and the center
    cos_separation = np.sin(np.radians(b_center)) * np.sin(np.radians(df['b'])) + np.cos(np.radians(b_center)) * np.cos(np.radians(df['b'])) * np.cos(np.radians(df['l'] - l_center))
    separation = np.degrees(np.arccos(cos_separation))
   
    # Filter the points based on the separation threshold
    mask = separation <= radius
    filtered_df = df[mask]
    filtered_df['g-i_color'] = filtered_df['gMeanPSFMag'] - filtered_df['iMeanPSFMag']
   
    # Plot the filtered points
    sc = ax.scatter(filtered_df['l'], filtered_df['b'], s=1, alpha=0.5, label='Filtered Pan-STARRS Stars', c=filtered_df['g-i_color'], cmap='inferno')
    customize_axes(ax)
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$g-i$')
   
    ax.set_xlabel('Galactic Longitude (l)')
    ax.set_ylabel('Galactic Latitude (b)')
    ax.set_title(f'Galactic Coordinate Plot - Zoomed in on {name}')
    
    # Set the x-axis and y-axis limits
    ax.set_xlim(l_center - radius, l_center + radius)
    ax.set_ylim(b_center - radius, b_center + radius)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/{name}_galactic_plot_zoomed.png')
    # plt.show()


def plot_hexbin_cmd(merged_df):
    set_plot_style()
    plt.figure(figsize=(10, 8))
    # Hexbin plot of g vs g-i for all data
    hb = plt.hexbin(merged_df['gMeanPSFMag'] - merged_df['iMeanPSFMag'], merged_df['gMeanPSFMag'], gridsize=150, cmap='gist_heat', bins='log')
    plt.xlabel('g - i')
    plt.ylabel('g')
    plt.gca().invert_yaxis()
    
    # Add colorbar to the hexbin plot
    cb = plt.colorbar(hb)
    cb.set_label('log(N)')
    customize_axes(plt.gca())
    plt.tight_layout()
    plt.savefig('hexbin_cmd.png', dpi=300)
    plt.close()
    
    set_plot_style()
    plt.figure(figsize=(10, 8))
    hb = plt.hexbin(merged_df['gMeanPSFMag'] - merged_df['rMeanPSFMag'], merged_df['gMeanPSFMag'], gridsize=150, cmap='gist_heat', bins='log')
    plt.xlabel('g - r')
    plt.ylabel('g')
    plt.title('Hexbin Plot: g vs g-r')
    plt.gca().invert_yaxis()  # Invert the y-axis

    cb = plt.colorbar(hb)
    cb.set_label('log(N)')
    customize_axes(plt.gca())
    plt.tight_layout()
    plt.savefig('gmag_g_r.png', dpi=300)
    plt.close()
    

    set_plot_style()
    plt.figure(figsize=(10, 8))
    plt.hexbin(merged_df['rMeanPSFMag'] - merged_df['iMeanPSFMag'], merged_df['rMeanPSFMag'], gridsize=150, cmap='gist_heat', bins='log')
    plt.xlabel('r - i')
    plt.ylabel('r')
    customize_axes(plt.gca())
    plt.gca().invert_yaxis()  # Invert the y-axis
    plt.tight_layout()
    plt.savefig('rmag_r_i.png', dpi=300)
    plt.close()
    
    # plt.figure(figsize=(10, 8))
    # plt.hexbin(merged_df['rMeanPSFMag'] - merged_df['zMeanPSFMag'], merged_df['rMeanPSFMag'], gridsize=150, cmap='gist_heat', bins='log')
    # plt.xlabel('r - z')
    # plt.ylabel('r')
    # plt.title('Hexbin Plot: r vs r-z')
    # plt.gca().invert_yaxis()  # Invert the y-axis
    # plt.tight_layout()
    # plt.savefig('hexbin_cmd4.png', dpi=300)
    # plt.close()
    
    # plt.figure(figsize=(10, 8))
    # plt.hexbin(merged_df['zMeanPSFMag'] - merged_df['yMeanPSFMag'], merged_df['zMeanPSFMag'], gridsize=150, cmap='gist_heat', bins='log')
    # plt.xlabel('z - y')
    # plt.ylabel('z')
    # plt.title('Hexbin Plot: z vs z-y')
    # plt.gca().invert_yaxis()  # Invert the y-axis
    # plt.tight_layout()
    # plt.savefig('hexbin_cmd5.png', dpi=300)
    # plt.close()
    
    # plt.figure(figsize=(10, 8))
    # plt.hexbin(merged_df['gMeanPSFMag'] - merged_df['zMeanPSFMag'], merged_df['gMeanPSFMag'], gridsize=150, cmap='gist_heat', bins='log')
    # plt.xlabel('g - z')
    # plt.ylabel('g')
    # plt.title('Hexbin Plot: g vs g-z')
    # plt.gca().invert_yaxis()  # Invert the y-axis
    # plt.tight_layout()
    # plt.savefig('hexbin_cmd6.png', dpi=300)
    # plt.close()
    
    # plt.figure(figsize=(10, 8))
    # plt.hexbin(merged_df['iMeanPSFMag'] - merged_df['yMeanPSFMag'], merged_df['iMeanPSFMag'], gridsize=150, cmap='gist_heat', bins='log')
    # plt.xlabel('i - y')
    # plt.ylabel('i')
    # plt.title('Hexbin Plot: i vs i-y')
    # plt.gca().invert_yaxis()  # Invert the y-axis
    # plt.tight_layout()
    # plt.savefig('hexbin_cmd7.png', dpi=300)
    # plt.close()
    
def plot_hexbin_colorcolor(merged_df):
    plt.figure(figsize=(10, 8))
    plt.hexbin(merged_df['gMeanPSFMag'] - merged_df['rMeanPSFMag'], merged_df['rMeanPSFMag'] - merged_df['iMeanPSFMag'], gridsize=150, cmap='gist_heat', bins='log')
    plt.xlabel('g - r')
    plt.ylabel('r - i')
    plt.title('Color-Color Diagram: g-r vs r-i')
    plt.tight_layout()
    plt.savefig('color_color_diagram.png', dpi=300)
    plt.close()

def plot_raerr_decerr(df):
    # Histogram of RAERR and DECERR
    fig, ax = plt.subplots(figsize=(10, 6))
    raerr = df['raMeanErr']*1000
    decerr = df['decMeanErr']*1000
    
    # Plot histogram of RAERR
    sns.histplot(raerr, bins=25000, ax=ax, color='blue', alpha=0.5, label=r'$\sigma_{\alpha}$')
    
    # Plot histogram of DECERR
    sns.histplot(decerr, bins=25000, ax=ax, color='red', alpha=0.5, label=r'$\sigma_{\delta}$')
    
    # Set the x-axis limits based on the 50th percentile (median) in arcseconds
    max_limit = max(raerr.quantile(0.5), decerr.quantile(0.5))
    ax.set_xlim(0, max_limit)
    
    # Set labels and title
    ax.set_xlabel('Error (milliarcseconds)')
    ax.set_ylabel('Count')
    ax.set_title(r'Histogram of $\sigma_{\alpha}$ and $\sigma_{\delta}$')
    
    # Adjust the layout and display the legend
    fig.tight_layout()
    ax.legend()
    plt.savefig('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/figures/raerr_decerr_hist.png', dpi=300)
    
    # Print statistics of RAERR and DECERR in arcseconds
    print("RAERR Statistics (milliarcseconds):")
    print(raerr.describe())
    print("\nDECERR Statistics (milliarcseconds):")
    print(decerr.describe())

def compute_galactic_coordinates(df):
    # Convert RA and Dec to numpy arrays
    ra = df['raMean'].values
    dec = df['decMean'].values

    # Create a SkyCoord object
    c = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame='icrs')

    # Convert to galactic coordinates
    galactic = c.galactic

    # Add the galactic coordinates to the dataframe
    df['l'] = galactic.l.deg
    df['b'] = galactic.b.deg
    # Plot the distribution of galactic coordinates
    plt.figure(figsize=(12, 8))
    plt.scatter(df['l'], df['b'], s=5, c='black', alpha=0.5)
    plt.xlabel('Galactic Longitude (l)')
    plt.ylabel('Galactic Latitude (b)')
    plt.title('Distribution of Galactic Coordinates')
    plt.tight_layout()
    plt.savefig('galactic_coordinates.png', format='png', dpi=300)
    
    return df


def plot_cmd_GC(filtered_data, filename, l_center, b_center, radius, name):
    
    cos_separation = np.sin(np.radians(b_center)) * np.sin(np.radians(filtered_data['b'])) + np.cos(np.radians(b_center)) * np.cos(np.radians(filtered_data['b'])) * np.cos(np.radians(filtered_data['l'] - l_center))
    separation = np.degrees(np.arccos(cos_separation))

    mask = separation <= radius
    data_to_plot = filtered_data[mask]

    if not data_to_plot.empty:
        plt.figure(figsize=(12, 12))
        plt.scatter(data_to_plot['gMeanPSFMag'] - data_to_plot['iMeanPSFMag'], data_to_plot['gMeanPSFMag'], c='black', s=30, alpha=0.8, marker='x')
        plt.xlabel('g - i')
        plt.ylabel('g')
        plt.title(f'Color-Magnitude Diagram (CMD) of {name} within {radius} degrees of (l={l_center}, b={b_center})')
        plt.gca().invert_yaxis()  # Invert the y-axis
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi=500)
        plt.close()
        
        # plot r vs g-r
        plt.figure(figsize=(12, 12))
        plt.scatter(data_to_plot['gMeanPSFMag'] - data_to_plot['rMeanPSFMag'], data_to_plot['rMeanPSFMag'], c='black', s=30, alpha=0.8, marker='x')
        plt.xlabel('g - r')
        plt.ylabel('r')
        plt.title(f'Color-Magnitude Diagram (CMD) of {name} within {radius} degrees of (l={l_center}, b={b_center})')
        plt.gca().invert_yaxis()  # Invert the y-axis
        plt.tight_layout()
        plt.savefig(f'r_{filename}', format='png', dpi=500)
        plt.close()
    else:
        print("No data points found within the specified radius.")
        
    

def plot_region_by_galactic_coordinates(data, band1, band2, mag_range, color_range, filename):
    color = f'{band1} - {band2}'
    data[color] = data[band1] - data[band2]

    mag_min, mag_max = mag_range
    color_min, color_max = color_range
    filtered_data = data[(data[band1] >= mag_min) & (data[band1] <= mag_max) &
                         (data[color] >= color_min) & (data[color] <= color_max)]

    if not filtered_data.empty:
        plt.figure(figsize=(12, 8))
        # plt.scatter(filtered_data['l'], filtered_data['b'], s=5, c='black', alpha=0.5)
        plt.hexbin(filtered_data['l'], filtered_data['b'], gridsize=150, cmap='gist_heat', bins='log')
        plt.xlabel('Galactic Longitude (l)')
        plt.ylabel('Galactic Latitude (b)')
        plt.title(f'Distribution of Stars in the Selected Region\n{band1} in [{mag_min}, {mag_max}], {color} in [{color_min}, {color_max}]')
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi=300)
    else:
        print("No data points found within the specified magnitude and color range.")

def deredden_magnitude(band, ebv, rv=3.1):
    """
    Apply reddening correction to a magnitude value in a given band.
   
    Parameters:
    band (str): The photometric band ('u','g', 'r', 'i', 'z', 'y', 'j', 'h', 'k').
    mag (float): The input magnitude value.
    ebv (float): The E(B-V) value.
    rv (float, optional): The ratio of total-to-selective extinction, R_V.
        3.1 for the Milky Way.
   
    Returns:
    float: The dereddened magnitude value.
    """
    coefficients_rv31 = {
        'psfMag_u': 4.239,
        'gMeanPSFMag': 3.172,
        'rMeanPSFMag': 2.271,
        'iMeanPSFMag': 1.682,
        'zMeanPSFMag': 1.322,
        'yMeanPSFMag': 1.087,
        'jmag': 0.709,
        'hmag': 0.449,
        'kmag': 0.302
    }
    if pd.isna(band):
        return np.nan
    
    if band not in coefficients_rv31:
        raise ValueError(f"Unsupported band: {band}")
   
    coeff = coefficients_rv31[band]
   
    return coeff * ebv

def plot_sdss_twomass_CMDs(df):
    plt.figure(figsize=(8, 6))
   
    mask = df['psfMag_u'].notna() & df['iMeanPSFMag'].notna()
    color = df.loc[mask, 'psfMag_u'] - df.loc[mask, 'iMeanPSFMag']
    magnitude = df.loc[mask, 'psfMag_u']
   
    plt.hexbin(color, magnitude, gridsize=150, cmap='viridis', mincnt=1)
    plt.gca().invert_yaxis()
    plt.xlabel('u - i')
    plt.ylabel('u')
    plt.title('Color-Magnitude Diagram')
    plt.colorbar(label='Count')
    plt.savefig('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/color_magnitude_diagram_u.png')
  
    plt.figure(figsize=(8, 6))
    mask = df['jmag'].notna() & df['hmag'].notna()
    color = df.loc[mask, 'jmag'] - df.loc[mask, 'hmag']
    magnitude = df.loc[mask, 'jmag']
   
    plt.hexbin(color, magnitude, gridsize=150, cmap='viridis', mincnt=1)
    plt.gca().invert_yaxis()
    plt.xlabel('j - h')
    plt.ylabel('j')
    plt.title('Color-Magnitude Diagram')
    plt.colorbar(label='Count')
    plt.savefig('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/color_magnitude_diagram_j.png')

if __name__ == '__main__':
    # Pan_Starrs_Phot = pd.read_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/PhotTable_bradhutc.csv')
    Pan_Starrs_Phot = pd.read_csv('/Users/bradhutc/Library/CloudStorage/OneDrive-Personal/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/PhotTable_bradhutc.csv')
    print(f'{len(Pan_Starrs_Phot)} Pan-STARRS Photometry entries loaded.')
    # Pan_Starrs_Pos=pd.read_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/MatchedTable_bradhutc.csv')
    Pan_Starrs_Pos = pd.read_csv('/Users/bradhutc/Library/CloudStorage/OneDrive-Personal/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/MatchedTable_bradhutc.csv')   
    Pan_Starrs = pd.merge(Pan_Starrs_Phot, Pan_Starrs_Pos, on='objID')
    bands = ['g', 'r', 'i', 'z', 'y']
    
    conditions = []
    for band in bands:
        mag_col = f'{band}MeanPSFMag'
        err_col = f'{band}MeanPSFMagErr'
        quality_col = f'{band}QfPerfect'
        num_meas = f'{band}MeanPSFMagNpt'
        
        conditions.append( 
                  (abs(Pan_Starrs[mag_col]) != 999) & 
                  (abs(Pan_Starrs[err_col]) != 999) & 
                  (abs(Pan_Starrs[quality_col]) != 999) & 
                  (Pan_Starrs[mag_col] < 24) & 
                  (Pan_Starrs[err_col] <= 0.05) & 
                  (Pan_Starrs[num_meas] >= 5) & 
                  (Pan_Starrs[quality_col] >= 0.99) & 
                  (Pan_Starrs['gMeanPSFMag'] - Pan_Starrs['rMeanPSFMag'] <= 2) & 
                  (Pan_Starrs['zMeanPSFMag'] - Pan_Starrs['yMeanPSFMag'] > -0.5) & 
                  (Pan_Starrs['zMeanPSFMag'] - Pan_Starrs['yMeanPSFMag'] < 1.0) & 
                  (Pan_Starrs['iMeanPSFMag'] - Pan_Starrs['yMeanPSFMag'] > -0.75)
                  )
        
    Pan_Starrs = Pan_Starrs[np.all(conditions, axis=0)]
    print(f'{len(Pan_Starrs)} Clean Stars PS loaded.')
    # SDSS_Phot = pd.read_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/MatchedTableSDSS_bradhutc.csv')[['objID', 'psfMag_u', 'psfMagErr_u']]
    # TWO_MASS_Phot = pd.read_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/MatchedTable2MASS_bradhutc.csv')[['objID', 'jmag', 'jmag_err', 'hmag', 'hmag_err', 'kmag', 'kmag_err']]
    SDSS_Phot = pd.read_csv('/Users/bradhutc/Library/CloudStorage/OneDrive-Personal/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/MatchedTableSDSS_bradhutc.csv')[['objID', 'psfMag_u', 'psfMagErr_u']]
    TWO_MASS_Phot = pd.read_csv('/Users/bradhutc/Library/CloudStorage/OneDrive-Personal/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/MatchedTable2MASS_bradhutc.csv')[['objID', 'jmag', 'jmag_err', 'hmag', 'hmag_err', 'kmag', 'kmag_err']]
    SDSS_Phot = SDSS_Phot[(SDSS_Phot['psfMag_u'] > 0) & (SDSS_Phot['psfMag_u'] < 26) & (SDSS_Phot['psfMagErr_u'] <= 0.05)
                           ]
    print(f'{len(SDSS_Phot)} Clean Stars loaded from SDSS.')
    
    # Drop any star in 2MASS that has a 'null' error measurement.
    TWO_MASS_Phot = TWO_MASS_Phot.dropna(subset=['jmag_err', 'hmag_err', 'kmag_err'])
    TWO_MASS_Phot = TWO_MASS_Phot[(TWO_MASS_Phot['jmag_err'] <= 0.05) & (TWO_MASS_Phot['hmag_err'] <= 0.05) & (TWO_MASS_Phot['kmag_err'] <= 0.05)]
    print(f'{len(TWO_MASS_Phot)} Clean Stars loaded from 2MASS.')
    # positions = pd.read_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/MatchedTable_bradhutc.csv')
    df = pd.merge(Pan_Starrs, SDSS_Phot, on='objID', how='left')
    df = pd.merge(df, TWO_MASS_Phot, on='objID', how='left')
    df['psfMag_u'].fillna(np.nan, inplace=True)  # Default missing 'psfMag_u' values to np.nan
    df['psfMagErr_u'].fillna(0, inplace=True)  # Default missing 'psfMagErr_u' values to 0
    df['jmag'].fillna(np.nan, inplace=True)  # Default missing 'j_m' values to np.nan
    df['hmag'].fillna(np.nan, inplace=True)  # Default missing 'hm' values to np.nan
    df['kmag'].fillna(np.nan, inplace=True)  # Default missing 'k_m' values to np.nan
    df = df[
    ((df['psfMag_u'].notna() & df['gMeanPSFMag'].notna() & (df['psfMag_u'] - df['gMeanPSFMag'] >= -1)) | df['psfMag_u'].isna() | df['gMeanPSFMag'].isna()) &
    ((df['jmag'].notna() & df['hmag'].notna() & (df['jmag'] - df['hmag'] <= 1)) | df['jmag'].isna() | df['hmag'].isna())
]
    print(f'{len(df)} Clean Stars loaded.')
    os.makedirs('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots', exist_ok=True)
    plot_raerr_decerr(df)
    # Convert RAERR and DECERR to arcseconds
    df['raerr_milliarcsec'] = df['raMeanErr'] * 1000
    df['decerr_milliarcsec'] = df['decMeanErr'] * 1000
    df = compute_galactic_coordinates(df)
    
    df = df[
        # (df['raerr_milliarcsec'] < 1000) & (df['decerr_milliarcsec'] < 1000) & 
            (df['gMeanPSFMag'] - df['iMeanPSFMag'] < 4) & (df['gMeanPSFMag'] - df['iMeanPSFMag'] > -1)]
    # Filter df based on RAERR and DECERR less than 10 arcseconds
    print(f'Filtered dataframe with RAERR and DECERR < 1000 milliarcseconds contains {len(df)} entries.')
    df['g-i'] = df['gMeanPSFMag'] - df['iMeanPSFMag']
    ngp_file = 'C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/SFD_dust_4096_ngp.fits'
    m = sfdmap.SFDMap(ngp_file)
    np.int = int
    # Get the RA and Dec values from the DataFrame
    ra = df['raMean'].values
    dec = df['decMean'].values

    # Query the SFD map to get the E(B-V) values for each RA and Dec pair
    ebv = np.zeros_like(ra)
    for i in range(len(ra)):
        ebv[i] = m.ebv(ra[i], dec[i])

    # Add the E(B-V) values as a new column in the DataFrame
    df['ebv'] = ebv
    # Apply reddening corrections using the E(B-V) values from the DataFrame
    df['psfMag_u'] = df.apply(lambda row: row['psfMag_u'] - deredden_magnitude('psfMag_u', row['ebv']), axis=1)
    df['gMeanPSFMag'] = df.apply(lambda row: row['gMeanPSFMag'] - deredden_magnitude('gMeanPSFMag', row['ebv']), axis=1)
    df['rMeanPSFMag'] = df.apply(lambda row: row['rMeanPSFMag'] - deredden_magnitude('rMeanPSFMag', row['ebv']), axis=1)
    df['iMeanPSFMag'] = df.apply(lambda row: row['iMeanPSFMag'] - deredden_magnitude('iMeanPSFMag', row['ebv']), axis=1)
    df['zMeanPSFMag'] = df.apply(lambda row: row['zMeanPSFMag'] - deredden_magnitude('zMeanPSFMag', row['ebv']), axis=1)
    df['yMeanPSFMag'] = df.apply(lambda row: row['yMeanPSFMag'] - deredden_magnitude('yMeanPSFMag', row['ebv']), axis=1)
    df['jmag'] = df.apply(lambda row: row['jmag'] - deredden_magnitude('jmag', row['ebv']), axis=1)
    df['hmag'] = df.apply(lambda row: row['hmag'] - deredden_magnitude('hmag', row['ebv']), axis=1)
    df['kmag'] = df.apply(lambda row: row['kmag'] - deredden_magnitude('kmag', row['ebv']), axis=1)
    
    df_grizy = df[['objID','raMean','decMean','l','b','gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag','gMeanPSFMagNpt', 'rMeanPSFMagNpt', 'iMeanPSFMagNpt', 'zMeanPSFMagNpt', 'yMeanPSFMagNpt', 'prob_Star'
                   , 'gMeanPSFMagErr', 'rMeanPSFMagErr', 'iMeanPSFMagErr', 'zMeanPSFMagErr', 'yMeanPSFMagErr']]
    df_ugrizy = df[['objID','raMean','decMean','l','b','psfMag_u', 'gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag']]
    df_ugrizyjhk = df[['objID','raMean','decMean','l','b','psfMag_u', 'gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag', 'jmag', 'hmag', 'kmag']]
    
    simbad_ready_to_match = df[(df['raerr_milliarcsec'] < 100) & (df['decerr_milliarcsec'] < 100)]
    print(f"Filtered merged dataframe with RAERR and DECERR < 100 milliarcseconds contains {len(simbad_ready_to_match)} entries. These are now ready to match to SIMBAD.")
    
    plot_mag_vs_err(df_grizy, 'C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/mag_vs_err_grizy.png')
    plot_sdss_twomass_CMDs(df)
    plot_hexbin_cmd(df_grizy)
    plot_hexbin_colorcolor(df_grizy)
    plot_b_vs_l_zoomed(df_grizy, l_center=42.21695, b_center=78.70685, radius=0.5, name='M3 Globular Cluster')
    plot_b_vs_l_zoomed(df_grizy, l_center=332.96299, b_center=79.76419, radius=0.5, name='M53 Globular Cluster')
    plot_b_vs_l_zoomed(df_grizy, l_center=42.15023, b_center=73.59225, radius=0.5, name='NGC 5466 Globular Cluster')
    plot_b_vs_l_zoomed(df_grizy, l_center=335.69874, b_center=78.94614, radius=0.5, name='NGC 5053 Globular Cluster')
    plot_b_vs_l_zoomed(df_grizy, l_center=252.847939, b_center = 77.188723, radius =0.5, name='NGC 4147 Globular Cluster')
    
    plot_b_vs_l(df_grizy)
    plot_band_measurements(df_grizy)
    plot_individual_histograms(df_grizy, 'C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/histograms')
    
    df_grizy = Pan_Starrs
    df_grizy = compute_galactic_coordinates(df_grizy)
    plot_cmd_GC(df_grizy, 'cmd_M3.png', 42.21695, 78.70685, 0.5, name='M3')
    plot_cmd_GC(df_grizy, 'cmd_M53.png', l_center=332.96299, b_center=79.76419, radius=0.5, name='M53')
    plot_cmd_GC(df_grizy, 'cmd_NGC5466.png', l_center=42.15023, b_center=73.59225, radius=0.5,name='NGC5466')
    plot_cmd_GC(df_grizy, 'cmd_NGC5053.png', l_center=335.69874, b_center=78.94614, radius=0.5,name='NGC5053')
    plot_cmd_GC(df_grizy, 'cmd_NGC4147.png', l_center=252.847939, b_center = 77.188723, radius =0.5, name='NGC4147')
    plot_region_by_galactic_coordinates(df_grizy,'gMeanPSFMag', 'iMeanPSFMag', (19.5, 20.5), (1, 1.5), 'region1.png')
    plot_region_by_galactic_coordinates(df_grizy,'gMeanPSFMag', 'iMeanPSFMag', (20, 22), (1.5, 2.5), 'region2.png')

    # simbad_ready_to_match.to_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/simbad_rtm.csv')
    df_grizy.to_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/PS_clean.csv', index=False)