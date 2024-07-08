import os
import speclite.filters
from matplotlib import pyplot as plt
import numpy as np

# Define the new directories
DATA_DIR = "/N/project/catypGC/Bradley/Data_ML"
PLOT_DIR = "/N/project/catypGC/Bradley/Plots_ML"


try:
    # Load the SDSS u and 2MASS JHK filters using speclite
    sdss_u = speclite.filters.load_filter('sdss2010-u')
    twomass = speclite.filters.load_filters('twomass-*')

    # Load the data from the table for PanSTARRS filters
    data_path = os.path.join(DATA_DIR, 'psfilters.txt')
    data = np.genfromtxt(data_path, skip_header=26)

    # Extract wavelengths and filter transmission values for PanSTARRS filters
    wavelengths_nm = data[:, 0]  # Assuming this is in nm
    wavelengths_angstroms = wavelengths_nm * 10  # Conversion factor: 1 nm = 10 Å
    ps1_g, ps1_r, ps1_i, ps1_z, ps1_y = data[:, 2], data[:, 3], data[:, 4], data[:, 5], data[:, 6]

    # Set up the figure and axes
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot SDSS u filter using speclite with color shading
    ax.plot(sdss_u.wavelength, sdss_u.response, label='SDSS u', color='blue')
    ax.fill_between(sdss_u.wavelength, sdss_u.response, alpha=0.2, color='blue')

    # Plot PanSTARRS filters (grizy) with color shading
    for filter_data, label, color in zip(
        [ps1_g, ps1_r, ps1_i, ps1_z, ps1_y],
        ['PanSTARRS g', 'PanSTARRS r', 'PanSTARRS i', 'PanSTARRS z', 'PanSTARRS y'],
        ['green', 'red', 'orange', 'purple', 'brown']
    ):
        ax.plot(wavelengths_angstroms, filter_data, label=label, color=color)
        ax.fill_between(wavelengths_angstroms, filter_data, alpha=0.2, color=color)

    # Plot 2MASS JHK filters using speclite with color shading
    for f, color in zip(twomass, ['darkgoldenrod', 'darkslategray', 'black']):
        ax.plot(f.wavelength, f.response * 1e4, label=rf'{f.name} * $10^4$', color=color)
        ax.fill_between(f.wavelength, f.response * 1e4, alpha=0.2, color=color)

    ax.set_ylim(0, 1)

    # Increase the size of tick lines and axis labels and numbers
    ax.tick_params(axis='both', which='major', labelsize=14, length=8, width=1.5)
    ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=16)
    ax.set_ylabel('Filter Response', fontsize=16)

    # Add legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()

    # Save the plot
    output_path = os.path.join(PLOT_DIR, 'filter_response_curves.png')
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to {output_path}")

    plt.show()

except FileNotFoundError as e:
    print(f"Error: File not found. {e}")
except ValueError as e:
    print(f"Error: Problem with data values. {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")