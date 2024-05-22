import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

def prepare_data(NN_file, matched_file):
    NN_data = pd.read_csv(NN_file)
    matched_data = pd.read_csv(matched_file)

    magnitudes = ['u', 'g', 'r', 'i', 'z']
    for mag in magnitudes:
        NN_data[mag] = NN_data[mag].round(2)
        matched_data[mag] = matched_data[mag].round(2)

    merged_data = pd.merge(NN_data, matched_data, on=magnitudes, how='inner')
    merged_data['g-i'] = merged_data['g'] - merged_data['i']
    NN_data['g-i'] = NN_data['g'] - NN_data['i']

    return NN_data, merged_data

def map_otype_to_numeric(data, column='object_type'):
    otype_mapping = {otype: i for i, otype in enumerate(data[column].unique())}
    data['OTYPE_numeric'] = data[column].map(otype_mapping)
    return data

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
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Object Type', fontsize='small')
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

def get_deviant_stars(data, band, threshold):
    """
    Identifies stars with a significant deviation between the original and reconstructed magnitude.

    Parameters:
    - data: DataFrame containing the star data.
    - band: The band of magnitude to check (e.g., 'u', 'g', 'r', 'i', 'z').
    - threshold: The magnitude difference threshold to consider a star significantly deviated.

    Returns:
    - A DataFrame with stars that have a significant deviation.
    """
    
    # Calculate the absolute difference between the original and reconstructed magnitude
    data[f'{band}_diff'] = abs(data[band] - data[f'reconstructed_{band}'])
    
    # Identify stars with a deviation greater than the threshold
    deviant_stars = data[data[f'{band}_diff'] > threshold]
    print(f'Found {len(deviant_stars)} stars with a deviation greater than {threshold} in the {band}-band magnitude.')
    
    return deviant_stars

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
    plt.savefig('poorreconstruction.png', format='png', dpi=300)
    plt.show()
def plot_advanced_color_conditions(data, filename='color_condition_plots.png'):
    data['g-i'] = data['g'] - data['i']
    data['g-r'] = data['g'] - data['r']

    # Define the color conditions
    conditions = [
        (data['g-i'] > 3),
        (data['g-r'] > 1.75)
    ]
    colors = ['red', 'blue']  # Colors for the conditions

    # Plot i vs g-r
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.scatter(data['g-r'], data['i'], color='grey', s=5, alpha=0.5)
    for condition, color in zip(conditions, colors):
        plt.scatter(data['g-r'][condition], data['i'][condition], color=color, s=5, label=f'{color} condition')
    plt.xlabel('g-r')
    plt.ylabel('i-band Magnitude')
    plt.title('i-band Magnitude vs. g-r')
    plt.legend()

    # Plot r vs g-r
    plt.subplot(1, 3, 2)
    plt.scatter(data['g-r'], data['r'], color='grey', s=5, alpha=0.5)
    for condition, color in zip(conditions, colors):
        plt.scatter(data['g-r'][condition], data['r'][condition], color=color, s=5, label=f'{color} condition')
    plt.xlabel('g-r')
    plt.ylabel('r-band Magnitude')
    plt.title('r-band Magnitude vs. g-r')
    plt.legend()

    # Plot latent space with color conditions
    plt.subplot(1, 3, 3)
    plt.scatter(data['Gamma'], data['Lambda'], color='grey', s=5, alpha=0.5) 
    for condition, color in zip(conditions, colors):
        plt.scatter(data['Gamma'][condition], data['Lambda'][condition], color=color, s=5, label=f'{color} condition')
    plt.xlabel('Gamma')
    plt.ylabel('Lambda')
    plt.title('Composite Stellar Feature Space with Color Conditions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=300)
    plt.close()

def plot_cmd(NN_data, filtered_data, filename, specific_types=None):
    plt.figure(figsize=(16, 12))
    if specific_types is not None:
        data_to_plot = filtered_data[filtered_data['object_type'].isin(specific_types)]
        palette = sns.color_palette("husl", n_colors=len(specific_types))
    else:
        data_to_plot = filtered_data
        palette = sns.color_palette("husl", n_colors=len(filtered_data['object_type'].unique()))
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', '+', 'x', '|', '_', 'P']
    marker_styles = {otype: markers[i % len(markers)] for i, otype in enumerate(data_to_plot['object_type'].unique())}

    plt.scatter(NN_data['g-i'], NN_data['i'], c='gray', s=0.2, alpha=0.3)
    for idx, (otype, group) in enumerate(data_to_plot.groupby('object_type')):
        plt.scatter(group['g-i'], group['i'], label=otype, marker=marker_styles[otype], color=palette[idx % len(palette)], s=30, alpha=0.9)

    plt.gca().invert_yaxis()
    plt.xlabel('g-i')
    plt.ylabel('i')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), title='Object Type', fontsize='small')
    plt.tight_layout()
    plt.savefig(filename, format='png', dpi=500)
    
    
if __name__ == '__main__':
    NN_data, merged_data = prepare_data('CSFS.csv', 'simbad_matched_cleaned.csv')
    merged_data = map_otype_to_numeric(merged_data)
    specific_types = ['BlueStraggler', 'WhiteDwarf', 'delSctV*', 'Variable*', 'RGB*', 'WhiteDwarf_Candidate', 'HotSubdwarf','RRLyrae']
    filtered_data = merged_data[merged_data['object_type'] != 'Star']
    filtered_data = map_otype_to_numeric(filtered_data, 'object_type')

    data = map_otype_to_numeric(merged_data, 'object_type')  # Map object types to numeric if necessary for filtering
    plot_cmd(NN_data, filtered_data, 'CMD_ALL.png', specific_types=specific_types)
    # plot_latent_space(NN_data, filtered_data, 'CSFS_all.png')
    plot_latent_space(NN_data, filtered_data, 'CSFS_specific_types.png', specific_types=specific_types)
    # plot_latent_space_by_magnitude(NN_data, 'latent_by_magnitude')
    # plot_latent_space_by_color_combinations(NN_data, 'latent_by_color_combinations')
    # plot_reconstruction_all_bands(NN_data)
    NN_data['g-r'] = NN_data['g'] - NN_data['r']
    NN_data['r-i'] = NN_data['r'] - NN_data['i']
    # num_deviant_stars = 0
    deviant_u_stars = get_deviant_stars(NN_data, 'u', threshold=0.5)
    deviant_g_stars = get_deviant_stars(NN_data, 'g', threshold=0.5)
    deviant_r_stars = get_deviant_stars(NN_data, 'r', threshold=0.5)
    deviant_i_stars = get_deviant_stars(NN_data, 'i', threshold=0.5)
    deviant_z_stars = get_deviant_stars(NN_data, 'z', threshold=0.5)
    
    plot_advanced_color_conditions(NN_data)
 
   
    
    # print(f'Found {num_deviant_stars} stars with a deviation greater than 0.2 in any band.')
    
    
    # plot_sample_stars(deviant_i_stars, sample_size=10, title='Stars with Deviant i-band Magnitudes')
    
    plt.figure(figsize=(12, 8))
    plt.scatter(NN_data['g-i'], NN_data['i'], c='gray', s=1, alpha=0.3, label = 'All Stars')
    plt.scatter(deviant_u_stars['g-i'], deviant_u_stars['i'], c='blue', s=5, alpha=0.7, label = 'Deviant u-band Stars')
    plt.scatter(deviant_g_stars['g-i'], deviant_g_stars['i'], c='green', s=5, alpha=0.7, label = 'Deviant g-band Stars')
    plt.scatter(deviant_r_stars['g-i'], deviant_r_stars['i'], c='red', s=5, alpha=0.7, label = 'Deviant r-band Stars')
    plt.scatter(deviant_i_stars['g-i'], deviant_i_stars['i'], c='orange', s=5, alpha=0.7, label = 'Deviant i-band Stars')
    plt.scatter(deviant_z_stars['g-i'], deviant_z_stars['i'], c='purple', s=5, alpha=0.7, label = 'Deviant z-band Stars')
    plt.gca().invert_yaxis()
    plt.xlabel('g-i Color')
    plt.ylabel('i-band Magnitude')
    plt.title('Stars with Deviant Reconstructed Magnitudes > 0.5')
    plt.tight_layout()
    plt.legend()
    plt.savefig('deviant_i_stars.png', format='png', dpi=300)
    plt.show()
    
    # now plot g-r vs r-i
    plt.figure(figsize=(12, 8))
    plt.scatter(NN_data['g-r'], NN_data['r-i'], c='gray', s=3, alpha=0.3, label = 'All Stars')
    plt.scatter(deviant_u_stars['g-r'], deviant_u_stars['r-i'], c='blue', s=5, alpha=0.7, label = 'Deviant u-band Stars')
    plt.scatter(deviant_g_stars['g-r'], deviant_g_stars['r-i'], c='green', s=5, alpha=0.7, label = 'Deviant g-band Stars')
    plt.scatter(deviant_r_stars['g-r'], deviant_r_stars['r-i'], c='red', s=5, alpha=0.7, label = 'Deviant r-band Stars')
    plt.scatter(deviant_i_stars['g-r'], deviant_i_stars['r-i'], c='orange', s=5, alpha=0.7, label = 'Deviant i-band Stars')
    plt.scatter(deviant_z_stars['g-r'], deviant_z_stars['r-i'], c='purple', s=5, alpha=0.7, label = 'Deviant z-band Stars')
    plt.xlabel('g-r')
    plt.ylabel('r-i')
    plt.title('Stars with Deviant Reconstructed Magnitudes > 0.5')
    plt.tight_layout()
    plt.legend()
    plt.savefig('deviant_i_stars_color.png', format='png', dpi=300)
    plt.show()
    