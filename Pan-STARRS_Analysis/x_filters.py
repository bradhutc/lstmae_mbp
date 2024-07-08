import speclite.filters
from matplotlib import pyplot as plt
import numpy as np

# Load the SDSS u and 2MASS JHK filters using speclite
sdss_u = speclite.filters.load_filter('sdss2010-u')
twomass = speclite.filters.load_filters('twomass-*')

# Load the data from the table for PanSTARRS filters
data = np.genfromtxt('C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/psfilters.txt',
                     skip_header=26)

# Extract wavelengths and filter transmission values for PanSTARRS filters
wavelengths = data[:, 0]
# Convert PanSTARRS wavelengths from nm to Angstroms
wavelengths_nm = data[:, 0]  # Assuming this is in nm
wavelengths_angstroms = wavelengths_nm * 10  # Conversion factor: 1 nm = 10 Å
ps1_g = data[:, 2]
ps1_r = data[:, 3]
ps1_i = data[:, 4]
ps1_z = data[:, 5]
ps1_y = data[:, 6]

# Set up the figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Plot SDSS u filter using speclite with color shading
ax.plot(sdss_u.wavelength, sdss_u.response, label='SDSS u', color='blue')
ax.fill_between(sdss_u.wavelength, sdss_u.response, alpha=0.2, color='blue')

# Plot PanSTARRS filters (grizy) with color shading
ax.plot(wavelengths*10, ps1_g, label='PanSTARRS g', color='green')
ax.fill_between(wavelengths*10, ps1_g, alpha=0.2, color='green')
ax.plot(wavelengths*10, ps1_r, label='PanSTARRS r', color='red')
ax.fill_between(wavelengths*10, ps1_r, alpha=0.2, color='red')
ax.plot(wavelengths*10, ps1_i, label='PanSTARRS i', color='orange')
ax.fill_between(wavelengths*10, ps1_i, alpha=0.2, color='orange')
ax.plot(wavelengths*10, ps1_z, label='PanSTARRS z', color='purple')
ax.fill_between(wavelengths*10, ps1_z, alpha=0.2, color='purple')
ax.plot(wavelengths*10, ps1_y, label='PanSTARRS y', color='brown')
ax.fill_between(wavelengths*10, ps1_y, alpha=0.2, color='brown')

# Plot 2MASS JHK filters using speclite with color shading
for f, color in zip(twomass, ['darkgoldenrod', 'darkslategray', 'black']):
    ax.plot(f.wavelength, f.response * 1e4, label=rf'{f.name} * $10^4$', color=color)
    ax.fill_between(f.wavelength, f.response * 1e4, alpha=0.2, color=color)

ax.set_ylim(0,1)
# Increase the size of tick lines and axis labels and numbers
ax.tick_params(axis='both', which='major', labelsize=14, length=8, width=1.5)
ax.set_xlabel(r'Wavelength ($\AA$)', fontsize=16)
ax.set_ylabel('Filter Response', fontsize=16)
# Add legend
ax.legend(loc='upper right')

plt.tight_layout()
plt.savefig('filter_response_curves.png', dpi=300)
plt.show()