import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Define the new directories
PLOT_DIR = "/N/project/catypGC/Bradley/Plots_ML"

# Set the plot style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Define the range of absolute magnitudes
M_abs = np.arange(-5, 18.1, 0.1)

# Apparent magnitude of the Pan-STARRS g-band bright limit
m_app = 14.5

# Calculate distances (in parsecs)
distances = 10**((m_app - M_abs + 5) / 5)

# Create the plot
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(distances / 1000, M_abs, 'b-', linewidth=2, color='black')
ax.invert_yaxis()

# Set labels and title
ax.set_ylabel('Absolute Magnitude (M$_G$)', fontsize=14)
ax.set_xlabel('Distance (kpc)', fontsize=14)
ax.set_title('Absolute Magnitude vs Distance for Pan-STARRS g-band Bright Limit', fontsize=16)

# Customize tick labels
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(x, '.0f') if x >= 1 else format(x, '.1f')))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

# Customize grid
ax.grid(True, which='major', linestyle='-', alpha=0.7)
ax.grid(True, which='minor', linestyle=':', alpha=0.4)

# Add reference lines
reference_lines = [1, 7.94, 12.59, 19.95, 31.62, 50.12, 79.43]
for line in reference_lines:
    ax.axvline(x=line, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.text(line + 1, -0.2, f'{line} kpc', color='r', alpha=0.8, rotation=90, va='bottom', fontsize=10)

# Add GAIA limit line
ax.axvline(x=3, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(3.1, -0.2, '3 kpc', color='blue', alpha=0.8, rotation=90, va='bottom', fontsize=10)

# Add annotations
ax.text(0.95, 0.05, 'Pan-STARRS g-band\nbright limit: 14.5 mag', 
        transform=ax.transAxes, fontsize=10, va='bottom', ha='right', 
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', alpha=0.7))

plt.tight_layout()

# Save the plot
try:
    plt.savefig(os.path.join(PLOT_DIR, 'absmag_vs_distance.png'), dpi=300)
    print(f"Plot saved successfully to {os.path.join(PLOT_DIR, 'absmag_vs_distance.png')}")
except Exception as e:
    print(f"Error saving the plot: {e}")

plt.show()

print("Example values:")
for M in [-5, -4, -3, -2, -1, 0]:
    d = 10**((m_app - M + 5) / 5) / 1000
    print(f"Absolute Magnitude: {M:.1f}, Distance: {d:.2f} kpc")