import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
def plot_mag_vs_err(df):
    # Define the photometric bands and their corresponding colors
    bands = ['g', 'r', 'i', 'z', 'y']
    colors = ['green', 'red', 'navy', 'maroon', 'black']
    
    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    
    # Plot the magnitude vs. error for each band in separate subplots
    for band, color, ax in zip(bands, colors, axs):
        mag_col = f'{band}MeanPSFMag'
        err_col = f'{band}MeanPSFMagErr'
        
        # Remove rows with default value of -999 in the current band's magnitude or error column
        filtered_df = df[(df[mag_col] != -999) & (df[err_col] != -999)]
        
        ax.scatter(filtered_df[mag_col], filtered_df[err_col], color=color, alpha=0.9, s=0.5)
        ax.set_xlim(8, 23)
        ax.set_ylim(0, 0.5)
        ax.set_xlabel(f'{band}')
        ax.set_ylabel(rf'$\sigma_{band}$')
        ax.set_title(f'{band}-band')
    
    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.savefig('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/mag_vs_err.png')

    # plt.show()
def plot_g_mag_vs_err(df):
    # Define the color for the main plot
    color = 'green'

    # Create the main plot
    fig, ax = plt.subplots(figsize=(8, 6))
    mag_col = 'gMeanPSFMag'
    err_col = 'gMeanPSFMagErr'
    qf_col = 'gQfPerfect'

    # Remove rows with default value of -999 in the magnitude, error, or gQfPerfect column
    filtered_df = df[(df[mag_col] != -999) & (df[err_col] != -999) & (df[qf_col] != -999)]

    # Plot the magnitude vs. error for the g band
    scatter = ax.scatter(filtered_df[mag_col], filtered_df[err_col], c=filtered_df[qf_col], cmap='viridis', alpha=0.9, s=0.5)
    ax.set_xlim(8, 23)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel('g')
    ax.set_ylabel(r'$\sigma_g$')
    ax.set_title('g-band Magnitude vs. Error')

    # Add a colorbar for gQfPerfect
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('gQfPerfect')

    # Create an inset plot
    axin = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
    axin.scatter(filtered_df[mag_col], filtered_df[err_col], c=filtered_df[qf_col], cmap='viridis', alpha=0.9, s=0.5)
    axin.set_xlim(10, 13)
    axin.set_ylim(0, 0.05)
    axin.set_xlabel('g')
    axin.set_ylabel(r'$\sigma_g$')
    axin.set_title('Zoomed-in Region')
    ax.indicate_inset_zoom(axin)


    plt.tight_layout()
    plt.savefig('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/g_mag_vs_err_zoom.png')
    # plt.show()
def plot_b_vs_l(df):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='aitoff')
    
    # Convert l and b to radians
    l_rad = np.deg2rad(df['l'] - 180)  # Shift l by 180 degrees
    b_rad = np.deg2rad(df['b'])

    ax.scatter(l_rad, b_rad, s=1, alpha=0.5, color='black', label='Filtered Pan-STARRS Stars')
    
    ax.set_xlabel('Galactic Longitude (l)')
    ax.set_ylabel('Galactic Latitude (b)')

    
    # Add a grid
    ax.grid(True)
    ax.legend()
    plt.savefig('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/galactic_projection.png')
    # plt.show()



def plot_individual_histograms(df, output_dir):
    columns_to_plot = df.columns.drop('objID')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for column in columns_to_plot:
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
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(band_labels, measurements, color=colors, alpha=0.7)
    ax.set_xlabel('Photometric Band')
    ax.set_ylabel('Number of Measurements')
    
    # Add labels on top of each bar
    for i, v in enumerate(measurements):
        ax.text(i, v + 0.1, str(int(v)), ha='center')
    
    plt.tight_layout()
    plt.savefig('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/band_measurements.png', dpi=300)

def plot_b_vs_l_zoomed(df, l_center, b_center, name, radius=5):
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
    

    cbar = plt.colorbar(sc)
    cbar.set_label(r'$g-i$') 
    

    ax.set_xlabel('Galactic Longitude (l)')
    ax.set_ylabel('Galactic Latitude (b)')
    ax.set_title(f'Galactic Coordinate Plot - Zoomed in on {name}')

    ax.set_xlim(l_center - radius, l_center + radius)
    ax.set_ylim(b_center - radius, b_center + radius)
    
    # Add a grid
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/{name}_galactic_plot_zoomed.png')
    # plt.show()
    
def plot_g_r_r_i(df):
    plt.figure(figsize=(8, 6))
    plt.hexbin(df['gMeanPSFMag'] - df['rMeanPSFMag'], df['rMeanPSFMag'] - df['iMeanPSFMag'], gridsize=100, cmap='inferno')
    plt.xlabel('g-r')
    plt.ylabel('r-i')
    plt.title('Color-Color Diagram')
    plt.savefig('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/color_color_diagram.png')

def plot_g_g_i(df):
    plt.figure(figsize=(8, 6))
    plt.hexbin(df['gMeanPSFMag'] - df['iMeanPSFMag'], df['gMeanPSFMag'], gridsize=100, cmap='inferno')
    plt.xlabel('g-i')
    plt.ylabel('g')
    plt.title('Color-Magnitude Diagram')
    plt.savefig('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/color_magnitude_diagram.png')
    
if __name__ == '__main__':
    df = pd.read_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/Pan-Starrs_Data/MatchedTable_bradhutc.csv')
    # Mac
    # df = pd.read_csv('/Users/bradhutc/Library/CloudStorage/OneDrive-Personal/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/Pan-Starrs_Data/MatchedTable_bradhutc.csv')
    print(f'{len(df)} Stars loaded.')
    os.makedirs('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots', exist_ok=True)
    # plot_mag_vs_err(df)
    bands = ['g', 'r', 'i', 'z', 'y']
    
    for band in bands:
        mag_col = f'{band}MeanPSFMag'
        err_col = f'{band}MeanPSFMagErr'
        err_col = f'{band}MeanPSFMagErr'
        quality_col = f'{band}QfPerfect'
        
        df = df[(df[mag_col] != -999) & (df[err_col] != -999)]
        df_clean = df[(df[err_col] <= 0.05) & (round(df[quality_col],2) == 1.0)]
        
    print(f'{len(df)} Stars remaining after filtering.')
    
    # plot_b_vs_l_zoomed(df, l_center=42.21695, b_center=78.70685, radius=1, name='M3 Globular Cluster')
    # plot_b_vs_l_zoomed(df, l_center=332.96299, b_center=79.76419, radius=1, name='M53 Globular Cluster')
    # plot_b_vs_l_zoomed(df, l_center=42.15023, b_center=73.59225, radius=1, name='NGC 5466 Globular Cluster')
    # plot_b_vs_l_zoomed(df, l_center=335.69874, b_center=78.94614, radius=1, name='NGC 5053 Globular Cluster')

    # plot_mag_vs_err(df)
    # # Call the plot_b_vs_l function
    # plot_g_mag_vs_err(df)
    # plot_b_vs_l(df)
    # plot_band_measurements(df)
    # plot_individual_histograms(df, 'C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/histograms')
    plot_g_r_r_i(df)
    plot_g_g_i(df)
    
    
    # df.to_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/Pan-Starrs_Data/PS_filtered.csv', index=False)
    # df_clean.to_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/Pan-Starrs_Data/PS_clean.csv', index=False)