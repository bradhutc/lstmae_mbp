import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgba_array
from matplotlib.ticker import AutoMinorLocator
from matplotlib.patches import Ellipse

import seaborn as sns

from itertools import combinations

import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as u


from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV

import scipy.stats as stats
from scipy.stats import gaussian_kde

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

def plot_latent_space(NN_data, filtered_data, filename, specific_types=None, max_labels=5):
    set_plot_style()
    
    if specific_types is not None:
        data_to_plot = filtered_data[filtered_data['object_type'].isin(specific_types)]
    else:
        data_to_plot = filtered_data
    
    unique_types = data_to_plot['object_type'].unique()
    num_batches = (len(unique_types) + max_labels - 1) // max_labels
    
    for batch in range(num_batches):
        start_idx = batch * max_labels
        end_idx = min((batch + 1) * max_labels, len(unique_types))
        
        fig, ax = plt.subplots()
        
        ax.scatter(NN_data['Gamma'], NN_data['Lambda'], c='gray', s=1, alpha=0.1, marker='.')
        
        palette = sns.color_palette("husl", n_colors=end_idx - start_idx)
        for idx, otype in enumerate(unique_types[start_idx:end_idx], start=start_idx):
            group = data_to_plot[data_to_plot['object_type'] == otype]
            ax.scatter(group['Gamma'], group['Lambda'], label=otype, color=palette[idx - start_idx], s=25, alpha=0.8, marker='x')
        
        ax.invert_yaxis()
        ax.invert_xaxis()
        customize_axes(ax)
        
        ax.set_xlabel(r'$\Gamma$', fontweight='bold')
        ax.set_ylabel(r'$\Lambda$', fontweight='bold')
        ax.set_title(f'Latent Space - Batch {batch + 1}', fontweight='bold')
        
        ax.legend(loc='upper right', frameon=True, framealpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f'{filename}_batch{batch + 1}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_latent_space_by_magnitude(data, filename_prefix):
    set_plot_style()
    
    bands = ['gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag']
    for band in bands:
        fig, ax = plt.subplots()
        scatter = ax.scatter(data['Gamma'], data['Lambda'], c=data[band], s=5, cmap='viridis', alpha=0.7, marker='x')
        ax.invert_yaxis()
        ax.invert_xaxis()
        customize_axes(ax)
        
        fig.colorbar(scatter, label=f'{band}')
        ax.set_xlabel(r'$\Gamma$', fontweight='bold')
        ax.set_ylabel(r'$\Lambda$', fontweight='bold')
        ax.set_title(f'Composite Stellar Feature Space (CSFS) Colored by {band}', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{filename_prefix}_{band}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_latent_space_by_color_combinations(data, filename_prefix):
    set_plot_style()
    
    bands = ['gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag']
    color_combinations = list(combinations(bands, 2))
   
    for combo in color_combinations:
        color_diff = data[combo[0]] - data[combo[1]]
        fig, ax = plt.subplots()
        scatter = ax.scatter(data['Gamma'], data['Lambda'], c=color_diff, s=5, cmap='gist_heat', alpha=0.7)
        ax.invert_yaxis()
        ax.invert_xaxis()
        customize_axes(ax)
        
        fig.colorbar(scatter, label=f'{combo[0]}-{combo[1]} Color')
        ax.set_xlabel(r'$\Gamma$', fontweight='bold')
        ax.set_ylabel(r'$\Lambda$', fontweight='bold')
        ax.set_title(f'CSFS Colored by {combo[0]}-{combo[1]} Color', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{filename_prefix}_{combo[0]}-{combo[1]}.png', dpi=300, bbox_inches='tight')
        plt.close()

def plot_cmd(data, data_simbad, band1, band2, filename_prefix):
    set_plot_style()
    
    color = f'{band1} - {band2}'
    data[color] = data[band1] - data[band2]
    data_simbad[color] = data_simbad[band1] - data_simbad[band2]

    fig, ax = plt.subplots()
    ax.scatter(data[color], data[band1], color='gray', s=1, alpha=0.3)
    ax.invert_yaxis()
    customize_axes(ax)
    
    ax.set_xlabel(color, fontweight='bold')
    ax.set_ylabel(band1, fontweight='bold')
    ax.set_title('Color-Magnitude Diagram (CMD) - No Labels', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_no_labels.png', dpi=300, bbox_inches='tight')
    plt.close()

    labels_to_plot = ['BlueStraggler', 'HorBranch*_Candidate', 'RRLyrae', 'RSCVnV*', 'HotSubdwarf']
    fig, ax = plt.subplots()

    for label in labels_to_plot:
        mask = data_simbad['object_type'] == label
        if mask.any():
            ax.scatter(data_simbad.loc[mask, color], data_simbad.loc[mask, band1], s=5, alpha=0.9, label=label, marker='x')

    ax.invert_yaxis()
    customize_axes(ax)
    
    ax.set_xlabel(color, fontweight='bold')
    ax.set_ylabel(band1, fontweight='bold')
    ax.set_title('Color-Magnitude Diagram (CMD) - With Labels', fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_with_labels.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_cmd_density(data, band1, band2, filename):
    set_plot_style()
    
    color = f'{band1} - {band2}'
    data[color] = data[band1] - data[band2]

    x = data[color]
    y = data[band1]
    xy = np.vstack([x, y])
    z = gaussian_kde(xy)(xy)

    fig, ax = plt.subplots()
    scatter = ax.scatter(x, y, c=z, s=5, edgecolor='none', alpha=0.5, cmap='gist_heat')
    ax.invert_yaxis()
    customize_axes(ax)
    
    fig.colorbar(scatter, label='Density')
    ax.set_xlabel(color, fontweight='bold')
    ax.set_ylabel(band1, fontweight='bold')
    ax.set_title('Color-Magnitude Diagram (CMD) - Density Plot', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_latent_space_by_category(NN_data, filtered_data, filename_prefix):
    set_plot_style()
   
    filtered_data['category'] = filtered_data['object_type'].map(object_type_categories)
    categories = ['Compact Object', 'Main Sequence', 'He Core', 'RGB', 'Post-RGB']
   
    category_colors = {
        'Compact Object': 'purple',
        'Main Sequence': 'blue',
        'He Core': 'green',
        'RGB': 'orange',
        'Post-RGB': 'red'
    }
   
    # Plot selected categories together
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(NN_data['Gamma'], NN_data['Lambda'], c='gray', s=1, alpha=0.1, marker='.', label='All Stars')
   
    for category in categories:
        category_data = filtered_data[filtered_data['category'] == category]
        ax.scatter(category_data['Gamma'], category_data['Lambda'],
                   label=category, s=25, alpha=0.8, 
                   c=category_colors[category], marker='x')
   
    ax.invert_yaxis()
    ax.invert_xaxis()
    customize_axes(ax)
    ax.set_xlabel(r'$\Gamma$', fontweight='bold')
    ax.set_ylabel(r'$\Lambda$', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_selected_categories.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot each category separately with object types
    for category in categories:
        fig, ax = plt.subplots(figsize=(12, 10))
       
        ax.scatter(NN_data['Gamma'], NN_data['Lambda'], c='gray', s=1, alpha=0.1, marker='.', label='All Stars')
       
        category_data = filtered_data[filtered_data['category'] == category]
        object_types = category_data['object_type'].unique()
        
        # Create a color map for object types
        n_colors = len(object_types)
        colors = plt.cm.rainbow(np.linspace(0, 1, n_colors))
        color_dict = dict(zip(object_types, colors))
        
        for object_type in object_types:
            object_type_data = category_data[category_data['object_type'] == object_type]
            ax.scatter(object_type_data['Gamma'], object_type_data['Lambda'],
                       label=object_type, s=25, alpha=0.8, 
                       c=[color_dict[object_type]], marker='x')
       
        ax.invert_yaxis()
        ax.invert_xaxis()
        customize_axes(ax)
       
        ax.set_xlabel(r'$\Gamma$', fontweight='bold')
        ax.set_ylabel(r'$\Lambda$', fontweight='bold')
        ax.set_title(f'Category: {category}')
       
        ax.legend(loc='upper right', frameon=True, framealpha=0.8)
       
        plt.tight_layout()
        plt.savefig(f'{filename_prefix}_{category.replace(" ", "_")}_object_types.png', dpi=300, bbox_inches='tight')
        plt.close()

# def plot_color_color_by_category(NN_data, filtered_data, filename_prefix):
#     set_plot_style()
   
#     filtered_data['category'] = filtered_data['object_type'].map(object_type_categories)
#     categories = ['Compact Object', 'Main Sequence', 'He Core', 'RGB', 'Post-RGB']
   
#     # Define the color scheme
#     category_colors = {
#         'Compact Object': 'purple',
#         'Main Sequence': 'blue',
#         'He Core': 'green',
#         'RGB': 'orange',
#         'Post-RGB': 'red'
#     }
   
#     # Plot selected categories together
#     fig, ax = plt.subplots(figsize=(12, 10))
#     ax.scatter(NN_data['gMeanPSFMag'] - NN_data['iMeanPSFMag'], NN_data['rMeanPSFMag'] - NN_data['gMeanPSFMag'], c='gray', s=1, alpha=0.6, marker='.', label='All Stars')
   
#     for category in categories:
#         category_data = filtered_data[filtered_data['category'] == category]
#         ax.scatter(category_data['gMeanPSFMag'] - category_data['iMeanPSFMag'], category_data['rMeanPSFMag'] - category_data['gMeanPSFMag'],
#                    label=category, s=25, alpha=0.8, c=category_colors[category], marker='x')
   
#     ax.invert_yaxis()
#     customize_axes(ax)
#     ax.set_xlabel(r'$g-i$', fontweight='bold')
#     ax.set_ylabel(r'$u-g$', fontweight='bold')
#     ax.legend(loc='upper right', frameon=True, framealpha=0.8)
#     plt.tight_layout()
#     plt.savefig(f'{filename_prefix}_selected_categories.png', dpi=300, bbox_inches='tight')
#     plt.close()

#     # Plot each category separately with object types
#     for category in categories:
#         fig, ax = plt.subplots(figsize=(12, 10))
       
#         ax.scatter(NN_data['gMeanPSFMag'] - NN_data['iMeanPSFMag'], NN_data['rMeanPSFMag'] - NN_data['gMeanPSFMag'], c='gray', s=1, alpha=0.6, marker='.', label='All Stars')
       
#         category_data = filtered_data[filtered_data['category'] == category]
#         object_types = category_data['object_type'].unique()
        
#         # Create a color map for object types
#         n_colors = len(object_types)
#         colors = plt.cm.rainbow(np.linspace(0, 1, n_colors))
#         color_dict = dict(zip(object_types, colors))
        
#         for object_type in object_types:
#             object_type_data = category_data[category_data['object_type'] == object_type]
#             ax.scatter(object_type_data['gMeanPSFMag'] - object_type_data['iMeanPSFMag'], 
#                        object_type_data['rMeanPSFMag'] - object_type_data['gMeanPSFMag'],
#                        label=object_type, s=25, alpha=0.8, c=[color_dict[object_type]], marker='x')
       
#         ax.invert_yaxis()
#         customize_axes(ax)
       
#         ax.set_xlabel(r'$g-i$', fontweight='bold')
#         ax.set_ylabel(r'$u-g$', fontweight='bold')
#         ax.set_title(f'Category: {category}')
       
#         ax.legend(loc='upper right', frameon=True, framealpha=0.8)
       
#         plt.tight_layout()
#         plt.savefig(f'{filename_prefix}_{category.replace(" ", "_")}_object_types.png', dpi=300, bbox_inches='tight')
#         plt.close()


def plot_latent_space_multiple_GCs(df_CSFS_grizy, filename_prefix, gc_data):
    set_plot_style()
    # plt.style.use('dark_background')

    # Combined plot
    fig, ax = plt.subplots(figsize=(16, 12))
    
    # Plot all points in the background
    ax.scatter(df_CSFS_grizy['Gamma'], df_CSFS_grizy['Lambda'], c='lightgrey', s=1, alpha=0.1, marker='.')
    
    # Define a color palette for the globular clusters (darker colors)
    colors = [
    "#00FFFF",  # Cyan
    "#FF1493",  # Deep Pink
    "#32CD32",  # Lime Green
    "#FFD700",  # Gold
    "#8A2BE2"   # Blue Violet
]
    
    # Define different markers for each globular cluster
    markers = ['o', 's', '^', 'D', 'v']
    
    for (gc_name, l_center, b_center, radius), color, marker in zip(gc_data, colors, markers):
        # Select stars within the radius of the globular cluster
        gc_stars = df_CSFS_grizy[
            (df_CSFS_grizy['l'] - l_center)**2 + (df_CSFS_grizy['b'] - b_center)**2 <= radius**2
        ]
        
        # Plot the globular cluster stars on the combined plot
        ax.scatter(gc_stars['Gamma'], gc_stars['Lambda'], c=color, s=25, alpha=0.8, 
                   marker=marker, label=gc_name, edgecolors='black', linewidths=0.5)
        
        # Create a separate plot for this globular cluster
        fig_single, ax_single = plt.subplots(figsize=(12, 9))
        ax_single.scatter(df_CSFS_grizy['Gamma'], df_CSFS_grizy['Lambda'], c='lightgrey', s=1, alpha=0.1, marker='.')
        ax_single.scatter(gc_stars['Gamma'], gc_stars['Lambda'], c=color, s=15, alpha=0.7, 
                          marker=marker, label=gc_name, edgecolors='black', linewidths=0.2)
        
        ax_single.set_xlim(0.3, 0.7)
        ax_single.set_ylim(0.3, 3.5)
    
        ax_single.invert_yaxis()
        ax_single.invert_xaxis()
        customize_axes(ax_single)
        ax_single.set_xlabel(r'$\Gamma$', fontweight='bold')
        ax_single.set_ylabel(r'$\Lambda$', fontweight='bold')
        ax_single.set_title(f'CSFS - {gc_name} Globular Cluster', fontweight='bold')
        ax_single.legend(loc='upper right', frameon=True, framealpha=0.8)
        
        plt.tight_layout()
        plt.savefig(f'{filename_prefix}_{gc_name.replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_single)
    
    # Finalize and save the combined plot
    ax.set_xlim(0.3,0.7)
    ax.set_ylim(0.3,3.5)
    ax.invert_yaxis()
    ax.invert_xaxis()
    customize_axes(ax)
    ax.set_xlabel(r'$\Gamma$', fontweight='bold')
    ax.set_ylabel(r'$\Lambda$', fontweight='bold')
    ax.set_title('Composite Stellar Feature Space (CSFS) - Globular Clusters', fontweight='bold')
    ax.legend(loc='upper right', frameon=True, framealpha=0.8)
    
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_combined.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_gmm(X, gmm, labels, title, filename):
    plt.figure(figsize=(12, 10))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7, s=5)
    
    for i, (mean, covar) in enumerate(zip(gmm.means_, gmm.covariances_)):
        v, w = np.linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / np.linalg.norm(w[0])
        angle = np.arctan2(u[1], u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = Ellipse(xy=mean, width=v[0], height=v[1], angle=180. + angle, 
                      edgecolor='red', facecolor='none', alpha=0.5)
        plt.gca().add_artist(ell)
    
    plt.title(title, fontweight='bold')
    plt.xlabel(r'$\Gamma$', fontweight='bold')
    plt.ylabel(r'$\Lambda$', fontweight='bold')
    plt.colorbar(label='Cluster')
    plt.gca().invert_yaxis()
    customize_axes(plt.gca())
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def gmm_clustering(data, n_components_range=range(2, 21)):
    # Use unscaled data
    X = data[['Gamma', 'Lambda']].values

    # Perform model selection using BIC
    bic = []
    for n_components in n_components_range:
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        gmm.fit(X)
        bic.append(gmm.bic(X))
    
    optimal_n_components = n_components_range[np.argmin(bic)]
    
    # Fit the optimal model
    gmm = GaussianMixture(n_components=optimal_n_components, random_state=42)
    labels = gmm.fit_predict(X)
    
    # Plot BIC scores
    plt.figure(figsize=(10, 6))
    plt.plot(n_components_range, bic, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('BIC')
    plt.title('BIC Score vs. Number of GMM Components')
    plt.savefig('gmm_bic_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot clustering results
    plot_gmm(X, gmm, labels, f'GMM Clustering (n_components={optimal_n_components})', 'gmm_clustering.png')
    
    return gmm, labels, X


def plot_latent_space_kde(data, filename):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 10))

    # Calculate the kernel density estimate
    x = data['Gamma']
    y = data['Lambda']
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    
    # Create a grid of points
    xi, yi = np.mgrid[x.min():x.max():500j, y.min():y.max():500j]
    zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
    
    # Plot the density map
    im = ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='gouraud', cmap='jet')
    # Add a colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Density', rotation=270, labelpad=15)
    
    ax.set_xlabel(r'$\Gamma$', fontweight='bold')
    ax.set_ylabel(r'$\Lambda$', fontweight='bold')
    ax.set_title('Latent Space Density (KDE)', fontweight='bold')
    
    ax.invert_yaxis()
    ax.invert_xaxis()
    customize_axes(ax)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_latent_space_within_and_beyond_threshold(data, threshold, filename):
    set_plot_style()
    # use dark mode plot
    # plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define the bands
    bands = [ 'gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag']

    # Create a mask for points within the threshold for all bands
    mask = np.ones(len(data), dtype=bool)
    for band in bands:
        mask &= np.abs(data[band] - data[f'{band}_prime']) <= threshold

    # Apply the mask to get the filtered data
    within_threshold = data[mask]
    beyond_threshold = data[~mask]

    # Plot the points beyond the threshold
    scatter_beyond = ax.scatter(beyond_threshold['Gamma'], beyond_threshold['Lambda'], 
                                c='gray', s=1, alpha=0.3, marker='x', label='Beyond threshold')

    # Plot the points within the threshold
    scatter_within = ax.scatter(within_threshold['Gamma'], within_threshold['Lambda'], 
                                c='deeppink', s=1, alpha=0.3, marker='x', label='Within threshold')

    ax.set_xlabel(r'$\Gamma$', fontweight='bold')
    ax.set_ylabel(r'$\Lambda$', fontweight='bold')
    ax.set_title(f'Latent Space - Points within and beyond {threshold} mag threshold', fontweight='bold')

    ax.invert_yaxis()
    ax.invert_xaxis()
    customize_axes(ax)

    ax.legend()

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    # Print the percentage of points that meet the criteria
    percentage_within = (len(within_threshold) / len(data)) * 100
    percentage_beyond = (len(beyond_threshold) / len(data)) * 100
    print(f"Percentage of points within {threshold} mag threshold: {percentage_within:.2f}%")
    print(f"Percentage of points beyond {threshold} mag threshold: {percentage_beyond:.2f}%")


def plot_mag_vs_reconstruction(df):
    set_plot_style()
    fig, ax = plt.subplots(figsize=(12, 10))

    # Define the bands
    bands = ['gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag']

    for band in bands:
        ax.scatter(df[band], df[f'{band}_prime'], s=1, label=band)
    ax.plot([0, 30], [0, 30], color='black', linestyle='--', label='1:1 line')
    ax.invert_yaxis()
    ax.set_xlabel('Original Magnitude', fontweight='bold')
    ax.set_ylabel('Reconstructed Magnitude', fontweight='bold')
    ax.set_title('Original vs. Reconstructed Magnitudes', fontweight='bold')

    ax.legend()
    customize_axes(ax)

    
    plt.tight_layout()
    plt.savefig('mag_vs_reconstruction.png', dpi=300, bbox_inches='tight')
    plt.close()
    
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


def plot_distance(df):
    # Plot histogram of 'distance' and 'distance_error'
    set_plot_style()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(df['distance'], bins=50, color='b', alpha=0.7)
    ax[0].set_xlabel('Distance (kpc)')
    ax[0].set_ylabel('Frequency')
    ax[0].set_title('Histogram of Distance')
    ax[1].hist(df['distance_error'], bins=50, color='r', alpha=0.7)
    ax[1].set_xlabel('Distance Error (kpc)')
    ax[1].set_ylabel('Frequency')
    
    customize_axes(ax[0])
    plt.savefig('distance_hist.png')
    plt.close()

    # Plot distance vs. distance_error
    set_plot_style()
    plt.figure(figsize=(8,6))
    plt.scatter(df['distance'], df['distance_error'], s=1)
    plt.xlabel('Distance (kpc)')
    plt.ylabel('Distance Error (kpc)')
    customize_axes(plt.gca())
    plt.savefig('distance_vs_distance_error.png')
    plt.close()

def compute_abs_magnitudes(df):
    df['G'] = df['gMeanPSFMag'] - 5*np.log10(df['distance']/10*1000)
    df['G_error'] = np.sqrt(df['gMeanPSFMagErr']**2 + (5/np.log(10)*df['distance_error']/df['distance'])**2)
    df['G_prime'] = df['gMeanPSFMag_prime'] - 5*np.log10(df['distance']/10*1000)
    df = df[df['G_error'] < 2]
    return df

def plot_h_r_diagram(df):
    # df = df[(df['gMeanPSFMag'] - df['gMeanPSFMag_prime'] < 0.1) & (df['rMeanPSFMag'] - df['rMeanPSFMag_prime'] < 0.1)]
    # Now plot it with G' vs g'-r'
    set_plot_style()
    plt.figure(figsize=(12,8))
    # plt.scatter(df['gMeanPSFMag_prime']-df['rMeanPSFMag_prime'], df['G_prime'], s=1, alpha=0.2, c='black', marker='o')
    plt.hexbin(df['gMeanPSFMag_prime']-df['rMeanPSFMag_prime'], df['G_prime'], gridsize=200, cmap='viridis', bins='log')
    # plt.hexbin(df['gMeanPSFMag_prime']-df['rMeanPSFMag_prime'], df['G_prime'], gridsize=100, cmap='gist_heat')
    plt.xlim(-0.6,1.8)
    plt.ylim(-6,18)
    plt.gca().invert_yaxis()
    plt.xlabel('g\'-r\'')
    plt.ylabel('G\'')
    customize_axes(plt.gca())
    # plt.grid()
    plt.savefig('hr_diagram_prime.png', dpi=250)
    plt.close()

    set_plot_style()
    plt.figure(figsize=(12,8))
    plt.hexbin(df['gMeanPSFMag']-df['rMeanPSFMag'], df['G'], gridsize=200, cmap='viridis', bins='log')
    # plt.errorbar(df['gMeanPSFMag']-df['rMeanPSFMag'], df['G'], xerr=np.sqrt(df['gMeanPSFMagErr']**2 + df['rMeanPSFMagErr']**2), yerr=df['G_error'], fmt='x', markersize=1, color='black', alpha=0.2)
    # plt.scatter(df['gMeanPSFMag']-df['rMeanPSFMag'], df['G'], s=10, alpha=0.5, c='blue', marker='x')
    # plt.hexbin(df['gMeanPSFMag']-df['rMeanPSFMag'], df['G'], gridsize=100, cmap='gist_heat')
    plt.xlim(-0.6,1.8)
    plt.ylim(-6,18)
    plt.gca().invert_yaxis()
    plt.xlabel('g-r')
    plt.ylabel('G')
    customize_axes(plt.gca())
    # plt.grid()
    plt.savefig('hr_diagram.png', dpi=250)
    plt.close()

    # Again but r-i
    set_plot_style()
    plt.figure(figsize=(12,8))
    plt.scatter(df['rMeanPSFMag_prime']-df['iMeanPSFMag_prime'], df['G'], s=1, alpha=0.2, c='black', marker='o')
    # plt.hexbin(df['rMeanPSFMag_prime']-df['iMeanPSFMag_prime'], df['G'], gridsize=100, cmap='gist_heat')
    plt.gca().invert_yaxis()
    plt.xlabel('r\'-i\'')
    plt.ylabel('G')
    customize_axes(plt.gca())
    # plt.grid()
    plt.savefig('hr_diagram_ri_prime.png', dpi=250)
    plt.close()
    




def color_latent_space_by_ABS_G(df, df_all):
    set_plot_style()
    plt.figure(figsize=(8,6))
    plt.scatter(df_all['Gamma'], df_all['Lambda'], c='gray', s=0.1, alpha=0.1)
    plt.scatter(df['Gamma'], df['Lambda'], c=df['G'], s=0.5)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.colorbar()
    plt.xlabel('Gamma')
    plt.ylabel('Lambda')
    customize_axes(plt.gca())
    plt.savefig('color_latent_space_by_abs_g.png')
    plt.close()

    # Also color by distance
    set_plot_style()
    plt.figure(figsize=(8,6))
    plt.scatter(df['Gamma'], df['Lambda'], c=df['distance'], s=0.5)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.colorbar()
    plt.xlabel('Gamma')
    plt.ylabel('Lambda')
    customize_axes(plt.gca())
    plt.savefig('color_latent_space_by_distance.png')
    plt.close()

def classify_gc_stars(df, gc_data):
    gc_class = pd.Series('Other', index=df.index)
    
    for _, l_center, b_center, radius in gc_data:
        # Select stars within the radius of the globular cluster
        gc_mask = (
            (df['l'] - l_center)**2 + (df['b'] - b_center)**2 <= radius**2
        )
        
        # Apply classifications only to stars within this globular cluster
        gc_class[gc_mask & (df['Gamma'] >= 0.5) & (df['Gamma'] < 0.53) & (df['Lambda'] >= 3) & (df['Lambda'] < 3.5)] = 'GC: MS'
        gc_class[gc_mask & (df['Gamma'] >= 0.46) & (df['Gamma'] < 0.5) & (df['Lambda'] >= 2.8) & (df['Lambda'] < 3.2)] = 'GC: MSTO'
        gc_class[gc_mask & (df['Gamma'] >= 0.35) & (df['Gamma'] < 0.46) & (df['Lambda'] >= 0.5) & (df['Lambda'] < 2.8)] = 'GC: RGB'
        gc_class[gc_mask & (df['Gamma'] >= 0.6) & (df['Gamma'] < 0.7) & (df['Lambda'] >= 1.7) & (df['Lambda'] < 2.3)] = 'GC: HB'
    
    return gc_class

def classify_stars_and_plot_latent_space(df_all, df_simbad, df_gaia, gc_data, filename_prefix):
    set_plot_style()

    # Classify Gaia stars based on HR diagram
    df_gaia = compute_abs_magnitudes(df_gaia)
    regions = {
        'Main Sequence': [(0.2, 1.5), (0, 16)],
        'Subgiant': [(0.5, 0.8), (2.5, 4.5)],
        'White Dwarf': [(-0.5, 0.5), (10, 16)],
    }
    
    df_gaia['HR_class'] = 'Other'
    for class_name, ((color_min, color_max), (mag_min, mag_max)) in regions.items():
        mask = (
            (df_gaia['gMeanPSFMag'] - df_gaia['rMeanPSFMag'] >= color_min) &
            (df_gaia['gMeanPSFMag'] - df_gaia['rMeanPSFMag'] <= color_max) &
            (df_gaia['G'] >= mag_min) &
            (df_gaia['G'] <= mag_max)
        )
        df_gaia.loc[mask, 'HR_class'] = class_name

    # Combine similar types for SIMBAD classifications (excluding Variable and Binary)
    combined_types = {
        'Main Sequence': ['Main Sequence', 'Main Sequence Turn', 'Lower Main Sequence', 'MainSequence*', 'SXPheV*', 'RotV*', 'ChemPec*', 'delSctV*', 'alf2CVnV*'],
        'Giant': ['RGB*', 'RGB*_Candidate'],
        'Subgiant': ['Subgiant'],
        'White Dwarf': ['White Dwarf', 'WhiteDwarf', 'WhiteDwarf_Candidate'],
        'Horizontal Branch': ['Blue Horizontal Branch', 'HorBranch*', 'HorBranch*_Candidate'],
        'RR Lyrae': ['RRLyrae', 'RRLyrae_Candidate'],
        'Blue Straggler': ['Blue Straggler', 'BlueStraggler'],
        'Compact Object': ['HotSubdwarf', 'HotSubdwarf_Candidate'],
    }
    
    df_simbad['Combined_class'] = df_simbad['object_type'].map(
        {subtype: combined_type for combined_type, subtypes in combined_types.items() for subtype in subtypes}
    ).fillna('Other')

    # Plot classifications in latent space
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Layer 1: All data in gray
    ax.scatter(df_all['Gamma'], df_all['Lambda'], c='lightgray', s=1, alpha=0.1, label='All Stars')
    
    # Create separate color palettes for each classification source
    gaia_palette = sns.color_palette("RdPu", n_colors=len(regions))
    simbad_palette = sns.color_palette("deep", n_colors=len(combined_types))
    gc_palette = sns.color_palette("bright", n_colors=len(gc_data))
    
    # Layer 2: Gaia classifications (circle marker)
    for i, (class_name, _) in enumerate(regions.items()):
        mask = df_gaia['HR_class'] == class_name
        color = gaia_palette[i]
        alpha = 0.3 if class_name == 'Main Sequence' else 0.3
        ax.scatter(df_gaia.loc[mask, 'Gamma'], df_gaia.loc[mask, 'Lambda'],
                   s=15, alpha=alpha, c=[color], label=f'Gaia: {class_name}', marker='o')
    
    # Layer 3: SIMBAD classifications (triangle marker)
    for i, class_name in enumerate(combined_types.keys()):
        mask = df_simbad['Combined_class'] == class_name
        ax.scatter(df_simbad.loc[mask, 'Gamma'], df_simbad.loc[mask, 'Lambda'],
                   s=40, alpha=0.7, c=[simbad_palette[i]], label=f'SIMBAD: {class_name}', 
                   marker='^', edgecolors='black', linewidth=0.5)

    # Layer 4: GC stars (star marker)
    for i, (gc_name, l_center, b_center, radius) in enumerate(gc_data):
        gc_mask = (
            (df_all['l'] - l_center)**2 + (df_all['b'] - b_center)**2 <= radius**2
        )
        ax.scatter(df_all.loc[gc_mask, 'Gamma'], df_all.loc[gc_mask, 'Lambda'],
                   s=50, alpha=0.8, c=[gc_palette[i]], label=f'GC: {gc_name}', marker='*', edgecolors='black')

    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel(r'$\Gamma$', fontsize=14)
    ax.set_ylabel(r'$\Lambda$', fontsize=14)
    ax.set_title('Latent Space with Combined Classifications', fontsize=16)
    ax.legend(markerscale=2, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    customize_axes(ax)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_latent_space_combined_classes.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot GC stars only
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.scatter(df_all['Gamma'], df_all['Lambda'], c='lightgray', s=1, alpha=0.1)
    for i, (gc_name, l_center, b_center, radius) in enumerate(gc_data):
        gc_mask = (df_all['l'] - l_center)**2 + (df_all['b'] - b_center)**2 <= radius**2
        ax.scatter(df_all.loc[gc_mask, 'Gamma'], df_all.loc[gc_mask, 'Lambda'],
                   s=50, alpha=0.8, c=[gc_palette[i]], label=f'GC: {gc_name}', marker='*', edgecolors='black')
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel(r'$\Gamma$', fontsize=14)
    ax.set_ylabel(r'$\Lambda$', fontsize=14)
    ax.set_title('Latent Space - Globular Clusters', fontsize=16)
    ax.legend(markerscale=2, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    customize_axes(ax)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_latent_space_GC.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot Gaia classifications only
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.scatter(df_all['Gamma'], df_all['Lambda'], c='lightgray', s=1, alpha=0.1)
    for i, (class_name, _) in enumerate(regions.items()):
        mask = df_gaia['HR_class'] == class_name
        color = gaia_palette[i]
        alpha = 0.3 if class_name == 'Main Sequence' else 0.3
        ax.scatter(df_gaia.loc[mask, 'Gamma'], df_gaia.loc[mask, 'Lambda'],
                   s=15, alpha=alpha, c=[color], label=f'Gaia: {class_name}', marker='o')
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel(r'$\Gamma$', fontsize=14)
    ax.set_ylabel(r'$\Lambda$', fontsize=14)
    ax.set_title('Latent Space - Gaia Classifications', fontsize=16)
    ax.legend(markerscale=2, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    customize_axes(ax)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_latent_space_Gaia.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Plot SIMBAD classifications only
    fig, ax = plt.subplots(figsize=(16, 14))
    ax.scatter(df_all['Gamma'], df_all['Lambda'], c='lightgray', s=1, alpha=0.1)
    for i, class_name in enumerate(combined_types.keys()):
        mask = df_simbad['Combined_class'] == class_name
        ax.scatter(df_simbad.loc[mask, 'Gamma'], df_simbad.loc[mask, 'Lambda'],
                   s=40, alpha=0.7, c=[simbad_palette[i]], label=f'SIMBAD: {class_name}', 
                   marker='^', edgecolors='black', linewidth=0.5)
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel(r'$\Gamma$', fontsize=14)
    ax.set_ylabel(r'$\Lambda$', fontsize=14)
    ax.set_title('Latent Space - SIMBAD Classifications', fontsize=16)
    ax.legend(markerscale=2, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    customize_axes(ax)
    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_latent_space_SIMBAD.png', dpi=300, bbox_inches='tight')
    plt.close()

    # Print statistics
    print("SIMBAD Classification statistics:")
    print(df_simbad['Combined_class'].value_counts(normalize=True) * 100)
    print("\nGaia Classification statistics:")
    print(df_gaia['HR_class'].value_counts(normalize=True) * 100)

def plot_cmd(df):
    set_plot_style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].hexbin(df['gMeanPSFMag']-df['iMeanPSFMag'], df['gMeanPSFMag'], gridsize=100, cmap='gist_heat')
    ax[0].set_xlabel('g')
    ax[0].set_ylabel('g-i')
    ax[0].set_title('g vs g-i')
    ax[1].hexbin(df['gMeanPSFMag_prime']-df['iMeanPSFMag_prime'], df['gMeanPSFMag_prime'], gridsize=100, cmap='gist_heat')
    ax[1].set_xlabel('g\'')
    ax[1].set_ylabel('g\'-i\'')
    ax[1].set_title('g\' vs g\'-i\'')
    ax[0].invert_yaxis()
    ax[1].invert_yaxis()
    customize_axes(ax[0])
    customize_axes(ax[1])
    plt.savefig('cmd.png')
    plt.close()

def plot_latent_space_by_distance(df, df_all):
    df = df[df['distance_error'] < 0.25]
    set_plot_style()
    fig, ax = plt.subplots(1, 4, figsize=(18, 6))
    for i, (distance_min, distance_max) in enumerate([(0, 0.5), (0.5, 1), (1, 2), (2, 3)]):
        mask = (df['distance'] >= distance_min) & (df['distance'] < distance_max)
        ax[i].scatter(df_all['Gamma'], df_all['Lambda'], c='lightgray', s=1, alpha=0.1, label='All Stars')
        ax[i].hexbin(df.loc[mask, 'Gamma'], df.loc[mask, 'Lambda'], gridsize=100, cmap='gist_heat', bins='log')
        ax[i].set_xlabel(r'$\Gamma$')
        ax[i].set_ylabel(r'$\Lambda$')
        ax[i].set_title(f'Distance: {distance_min}-{distance_max} kpc')
        ax[i].set_xlim(-0.5, 1.0)
        ax[i].set_ylim(0, 4)
        ax[i].invert_yaxis()
        ax[i].invert_xaxis()
        customize_axes(ax[i])
    plt.tight_layout()
    plt.savefig('latent_space_by_distance.png')
    plt.close()

    set_plot_style()
    plt.figure(figsize=(8,6))
    mask = round(df['distance'],2) == 1
    mask2 = round(df['distance'],2) == 0.5
    mask3 = round(df['distance'],2) == 2
    plt.scatter(df_all['Gamma'], df_all['Lambda'], c='lightgray', s=1, alpha=0.1, label='All Stars')
    plt.hexbin(df.loc[mask, 'Gamma'], df.loc[mask, 'Lambda'], gridsize=100, cmap='gist_heat', bins='log')
    plt.hexbin(df.loc[mask2, 'Gamma'], df.loc[mask2, 'Lambda'], gridsize=100, cmap='viridis', bins='log')
    plt.hexbin(df.loc[mask3, 'Gamma'], df.loc[mask3, 'Lambda'], gridsize=100, cmap='cividis', bins='log')
    plt.xlabel(r'$\Gamma$')
    plt.ylabel(r'$\Lambda$')
    plt.title('CSFS - Log Density Plots')
    plt.xlim(-0.5, 1.0)
    plt.ylim(0, 4)
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    customize_axes(plt.gca())
    plt.tight_layout()
    plt.savefig('latent_space_distance_1kpc.png')
    plt.close()
    
if __name__ == '__main__':
    object_type_categories = {
    'EclBin_Candidate': 'Binary', 'SB*': 'Binary', 'RSCVnV*': 'Binary', 'BYDraV*': 'Binary', 'EclBin': 'Binary',
    'WhiteDwarf_Candidate': 'Compact Object', 'HotSubdwarf_Candidate': 'Compact Object', 
    'HotSubdwarf': 'Compact Object', 'WhiteDwarf': 'Compact Object',
    'RRLyrae': 'He Core', 'HorBranch*_Candidate': 'He Core', 'HorBranch*': 'He Core', 
    'RRLyrae_Candidate': 'He Core',
    'Low-Mass*': 'ignore', 'Star': 'ignore',
    'SXPheV*': 'Main Sequence', 'MainSequence*': 'Main Sequence', 'RotV*': 'Main Sequence', 
    'ChemPec*': 'Main Sequence', 'delSctV*': 'Main Sequence', 'BlueStraggler': 'Main Sequence', 
    'alf2CVnV*': 'Main Sequence',
    'HighVel*': 'Pop II', 'HighPM*': 'Pop II',
    'LongPeriodV*_Candidate': 'Post-RGB', 'AGB*': 'Post-RGB', 'C*_Candidate': 'Post-RGB', 
    'C*': 'Post-RGB', 'LongPeriodV*': 'Post-RGB', 'Cepheid': 'Post-RGB',
    'RGB*_Candidate': 'RGB', 'RGB*': 'RGB',
    'PulsV*': 'Variable', 'Variable*': 'Variable'
    }

    
    classification_map = {
        'Main Sequence': ['Main Sequence Turn','Main Sequence', 'Lower Main Sequence', 'MainSequence*', 'SXPheV*', 'RotV*', 'ChemPec*', 'delSctV*', 'alf2CVnV*'],
        'Blue Straggler': ['Blue Straggler', 'BlueStraggler'],
        'Red Giant Branch': ['RGB*', 'RGB*_Candidate'],
        'Subgiant': ['Subgiant', 'Sub Giant'],
        'Blue Horizontal Branch': ['Blue Horizontal Branch', 'HorBranch*', 'HorBranch*_Candidate'],
        'White Dwarf': ['White Dwarfs', 'WhiteDwarf', 'WhiteDwarf_Candidate'],
        'Main Sequence Turn Off': [ 'Main Sequence Turn Off'],
        'RR Lyrae': ['RRLyrae', 'RRLyrae_Candidate'],
        'Compact Object': ['HotSubdwarf', 'HotSubdwarf_Candidate']
        }
    # df_CSFS_grizy = pd.read_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/CSFS_grizy.csv')
    df_CSFS_grizy = pd.read_csv('/Users/bradhutc/Library/CloudStorage/OneDrive-Personal/Astrophysics_Research/CSFS_grizy.csv')
    df_CSFS_err = pd.read_csv('/Users/bradhutc/Library/CloudStorage/OneDrive-Personal/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/PS_clean.csv')[['objID','gMeanPSFMagErr', 'rMeanPSFMagErr', 'iMeanPSFMagErr', 'zMeanPSFMagErr', 'yMeanPSFMagErr']]
    df_CSFS_grizy = pd.merge(df_CSFS_grizy, df_CSFS_err, left_on='objID', right_on='objID', how='inner')
    df_SIMBAD = pd.read_csv('/Users/bradhutc/Library/CloudStorage/OneDrive-Personal/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/simbad_cleaned.csv')[['objID', 'object_type']]
    # df_SIMBAD= pd.read_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/simbad_cleaned.csv')[['objID', 'object_type']]
    # Plots for CSFS_grizy
    grizy_SIMBAD = pd.merge(df_CSFS_grizy, df_SIMBAD, left_on='objID', right_on='objID', how='inner')
    filtered_data = grizy_SIMBAD[(grizy_SIMBAD['object_type'] != 'Star') & (grizy_SIMBAD['object_type'] != 'Low-Mass*')]
    
    # plot_latent_space_by_color_combinations(df_CSFS_grizy, 'CSFS_grizy_color_combinations')
    # plot_latent_space_by_magnitude(df_CSFS_grizy, 'CSFS_grizy_magnitudes')
    # plot_latent_space(df_CSFS_grizy, filtered_data, 'CSFS_grizy_all_types.png')

    # plot_latent_space_by_category(df_CSFS_grizy, filtered_data, 'CSFS_grizy_by_category')
    # # plot_color_color_by_category(df_CSFS_grizy, filtered_data, 'grizy_color_color_by_category')
    # plot_cmd(df_CSFS_grizy, grizy_SIMBAD, 'gMeanPSFMag', 'rMeanPSFMag', 'cmd_grizy')
    # plot_latent_space_by_magnitude(df_CSFS_grizy, 'CSFS_grizy_magnitudes')

    # Calculate colors for both datasets
    # df_CSFS_grizy['g_i'] = df_CSFS_grizy['gMeanPSFMag'] - df_CSFS_grizy['iMeanPSFMag']
    # df_selected['u_g'] = df_selected['psfMag_u'] - df_selected['gMeanPSFMag']
    # df_selected['g_i'] = df_selected['gMeanPSFMag'] - df_selected['iMeanPSFMag']

    gc_data = [
    ('M3', 42.21695, 78.70685, 0.25),
    ('M53', 332.96299, 79.76419, 0.25),
    ('NGC5466', 42.15023, 73.59225, 0.25),
    ('NGC5053', 335.69874, 78.94614, 0.25),
    ('NGC4147', 252.847939, 77.188723, 0.25)
    ]

    # # Usage
    # plot_latent_space_within_and_beyond_threshold(df_CSFS_grizy, 0.01, 'latent_space_within_and_beyond_0.01mag.png')
    # plot_latent_space_within_and_beyond_threshold(df_CSFS_grizy, 0.05, 'latent_space_within_and_beyond_0.05mag.png')
    # plot_latent_space_within_and_beyond_threshold(df_CSFS_grizy, 0.1, 'latent_space_within_and_beyond_0.1mag.png')
    # plot_latent_space_within_and_beyond_threshold(df_CSFS_grizy, 0.2, 'latent_space_within_and_beyond_0.2mag.png')
    # plot_latent_space_within_and_beyond_threshold(df_CSFS_grizy, 0.5, 'latent_space_within_and_beyond_0.5mag.png')

    plot_latent_space_multiple_GCs(df_CSFS_grizy, 'CSFS_grizy_GCs', gc_data)
    # Lets only plot stars where the true magnitude is within 0.1 mag of the predicted magnitude
    # df_CSFS_grizy = df_CSFS_grizy[
    #                                 ((df_CSFS_grizy['gMeanPSFMag'] - df_CSFS_grizy['gMeanPSFMag_prime']).abs() <= 0.1) &
    #                                 ((df_CSFS_grizy['rMeanPSFMag'] - df_CSFS_grizy['rMeanPSFMag_prime']).abs() <= 0.1) &
    #                                 ((df_CSFS_grizy['iMeanPSFMag'] - df_CSFS_grizy['iMeanPSFMag_prime']).abs() <= 0.1) &
    #                                 ((df_CSFS_grizy['zMeanPSFMag'] - df_CSFS_grizy['zMeanPSFMag_prime']).abs() <= 0.1) &
    #                                 ((df_CSFS_grizy['yMeanPSFMag'] - df_CSFS_grizy['yMeanPSFMag_prime']).abs() <= 0.1)]
    
    # filtered_data = pd.merge(df_CSFS_grizy, df_SIMBAD, left_on='objID', right_on='objID', how='inner')
    # plot_latent_space_by_category(df_CSFS_grizy, filtered_data, 'CSFS_grizy_by_category')
    # plot_mag_vs_reconstruction(df_CSFS_grizy)
    
    
    # df_GAIA = pd.read_csv('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/MatchedTableGaia_bradhutc.csv')[['objID','parallax', 'parallax_error']]
    df_GAIA = pd.read_csv('/Users/bradhutc/Library/CloudStorage/OneDrive-Personal/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/MatchedTableGaia_bradhutc.csv')[['objID','parallax', 'parallax_error']]
    # Only positive gaia parallaxes
    
    # df_GAIA = df_GAIA[df_GAIA['parallax'] >= 0]
    df = pd.merge(df_CSFS_grizy, df_GAIA, on='objID', how='inner')
    df['distance'] = 1/df['parallax'] #in kpc
    df['distance_error'] = df['parallax_error']/df['parallax']*df['distance'] #in kpc
    df = df[df['distance_error'] < 0.1]
    # only stars within 3 kpc
    df = df[df['distance'] < 3]
    # plot_distance(df)
    df = compute_abs_magnitudes(df)
    plot_h_r_diagram(df)
    print(f"Min G: {df['G'].min()}, Max G: {df['G'].max()}")
    # # color_latent_space_by_ABS_G(df, df_CSFS_grizy)
    # # classify_stars_and_plot_latent_space(df, df_CSFS_grizy, 'ClassifiedfromHR')
    # plot_cmd(df_CSFS_grizy)

    
    # classify_stars_and_plot_latent_space(df_CSFS_grizy, grizy_SIMBAD, df, gc_data, 'Combined_Classification')
    # plot_latent_space_by_distance(df, df_CSFS_grizy)