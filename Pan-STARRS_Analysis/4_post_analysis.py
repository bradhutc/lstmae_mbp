import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from itertools import combinations
import itertools
import numpy as np
def plot_latent_space(NN_data, filtered_data, filename, specific_types=None):
    plt.figure(figsize=(16, 12))
    if specific_types is not None:
        data_to_plot = filtered_data[filtered_data['object_type'].isin(specific_types)]
        palette = sns.color_palette("husl", n_colors=len(specific_types))
    else:
        data_to_plot = filtered_data
        palette = sns.color_palette("husl", n_colors=len(filtered_data['object_type'].unique()))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+', 'x', '|', '_', 'P']
    marker_styles = {otype: markers[i % len(markers)] for i, otype in enumerate(data_to_plot['object_type'].unique())}

    plt.scatter(NN_data['Gamma'], NN_data['Lambda'], c='gray', s=0.2, alpha=0.3)
    for idx, (otype, group) in enumerate(data_to_plot.groupby('object_type')):
        plt.scatter(group['Gamma'], group['Lambda'], label=otype, marker=marker_styles[otype], color=palette[idx % len(palette)], s=30, alpha=0.9)

    plt.xlabel('Gamma')
    plt.ylabel('Lambda')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Stellar Type', fontsize='small')
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=500)

def plot_latent_space_by_magnitude(data, filename_prefix):
    magnitudes = ['u', 'g', 'r', 'i', 'z']
    for mag in magnitudes:
        plt.figure(figsize=(12, 8))
        plt.scatter(data['Gamma'], data['Lambda'], c=data[mag], s=5, cmap='viridis', alpha=0.7)
        plt.colorbar(label=f'{mag}-band Magnitude')
        plt.xlabel('Gamma')
        plt.ylabel('Lambda')
        plt.title(f'Composite Stellar Feature Space (CSFS) Colored by {mag}-band Magnitude')
        plt.tight_layout()
        plt.savefig(f'{filename_prefix}_{mag}.png', format='png', dpi=300)
        plt.close()

def plot_latent_space_by_color_combinations(data, filename_prefix):
    magnitudes = ['u', 'g', 'r', 'i', 'z']
    color_combinations = list(combinations(magnitudes, 2))  # Get all unique pairs
    
    for combo in color_combinations:
        color_diff = data[combo[0]] - data[combo[1]]
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(data['Gamma'], data['Lambda'], c=color_diff, s=5, cmap='coolwarm', alpha=0.7)
        plt.colorbar(scatter, label=f'{combo[0]}-{combo[1]} Color')
        plt.xlabel('Gamma')
        plt.ylabel('Lambda')
        plt.title(f'Composite Stellar Feature Space (CSFS) Colored by {combo[0]}-{combo[1]} Color')
        plt.tight_layout()
        plt.savefig(f'{filename_prefix}_{combo[0]}-{combo[1]}.png', format='png', dpi=300)
        plt.close()

def plot_reconstruction_all_bands(data):
    bands = ['u', 'g', 'r', 'i', 'z']
    for band in bands:
        plt.figure(figsize=(12, 8))
        # Scatter plot for original vs reconstructed magnitudes
        plt.scatter(data[band], data[f'reconstructed_{band}'], color='black',s=1, alpha=0.3, label='Reconstructed vs Original')
        plt.xlabel(f'{band}-band Magnitude')
        plt.ylabel(f'Reconstructed {band}-band Magnitude')
        plt.title(f'Reconstruction of {band}-band Magnitude')

        # Adding a line of slope 1 for perfect reconstruction
        lims = [
            min(min(data[band]), min(data[f'reconstructed_{band}'])),  # min of both axes
            max(max(data[band]), max(data[f'reconstructed_{band}']))   # max of both axes
        ]
        plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0, label='Perfect Reconstruction', color='red')  # Line of slope 1
        plt.xlim(lims)
        plt.ylim(lims)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'reconstruction_{band}.png', format='png', dpi=300)
        plt.close()

def plot_latent_space_zoomed(merged_df, filtered_data, filename, l_center, b_center, radius=0.5):
    plt.figure(figsize=(16, 12))
    
    # Calculate the angular separation between each point and the center
    cos_separation = np.sin(np.radians(b_center)) * np.sin(np.radians(filtered_data['b'])) + np.cos(np.radians(b_center)) * np.cos(np.radians(filtered_data['b'])) * np.cos(np.radians(filtered_data['l'] - l_center))
    separation = np.degrees(np.arccos(cos_separation))
    
    # Filter the points based on the separation threshold
    mask = separation <= radius
    data_to_plot = filtered_data[mask]
    
    if not data_to_plot.empty:
        palette = sns.color_palette("husl", n_colors=len(data_to_plot['object_type'].unique()))
        markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+', 'x', '|', '_', 'P']
        marker_styles = {otype: markers[i % len(markers)] for i, otype in enumerate(data_to_plot['object_type'].unique())}

        plt.scatter(merged_df['Gamma'], merged_df['Lambda'], c='gray', s=0.2, alpha=0.3)
        for idx, (otype, group) in enumerate(data_to_plot.groupby('object_type')):
            plt.scatter(group['Gamma'], group['Lambda'], label=otype, marker=marker_styles[otype], color=palette[idx % len(palette)], s=30, alpha=0.9)

        plt.xlabel('Gamma')
        plt.ylabel('Lambda')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Stellar Type', fontsize='small')
        plt.tight_layout()
        plt.savefig(filename, format='png', dpi=500)
    else:
        print("No data points found within the specified radius.")


if __name__ == '__main__':
    df_CSFS = pd.read_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/CSFS_PS_Clean.csv')
    df_SIMBAD = pd.read_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/simbad_matched.csv')
   
    # Merge the dataframes based on 'objID'
    merged_df = pd.merge(df_CSFS, df_SIMBAD, left_on='objID', right_on='PS_id', how='inner')
   
    filtered_data = merged_df[merged_df['object_type'] != 'Star']
   
    # Get unique labels from the 'object_type' column
    # unique_labels = list(filtered_data['object_type'].unique())
   
    # # Initialize variables to keep track of plotted labels and combinations
    # plotted_labels = set()
    # plot_count = 1
   
    # # Iterate over unique labels and create plots
    # while unique_labels:
    #     # Get the next 5 labels to plot
    #     combo = unique_labels[:5]
    #     plotted_labels.update(combo)
    #     plot_latent_space(merged_df, filtered_data, f'latent_space_plot_{plot_count}.png', combo)
    #     plot_count += 1
       
    #     # Remove the plotted labels from the unique_labels list
    #     unique_labels = [label for label in unique_labels if label not in plotted_labels]
    
    
    # specific_types = ['BlueStraggler', 'HorBranch*_Candidate','BrownD*_Candidate','HotSubdwarf_Candidate', 'delSctV', 'Variable']
    # filtered_data = merged_df[merged_df['object_type'] != 'Star']
   
    # # Plot the latent space with specific types labeled
    # plot_latent_space(merged_df, filtered_data, 'latent_space_specific_types.png', specific_types)
    
    l_center = 42.15023
    b_center = 73.59225
    radius = 0.5
    
    # Filter the merged data based on the angular separation from the cluster center
    cos_separation = np.sin(np.radians(b_center)) * np.sin(np.radians(merged_df['b'])) + np.cos(np.radians(b_center)) * np.cos(np.radians(merged_df['b'])) * np.cos(np.radians(merged_df['l'] - l_center))
    separation = np.degrees(np.arccos(cos_separation))
    mask = separation <= radius
    filtered_data = merged_df[mask]
    
    # Plot the latent space with stars within the specified radius and their labels
    plot_latent_space_zoomed(merged_df, filtered_data, 'latent_space_zoomed.png', l_center, b_center, radius)