import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Set the plot style
# plt.style.use('seaborn-darkgrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# Define the range of absolute magnitudes
M_abs = np.arange(-5, 18.1, 0.1)

#  Apparent magnitude of the Pan-STARRS g-band bright limit
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

# Set scales and limits
# ax.set_xscale('log')
# ax.set_xlim(0.1, 80)
# ax.set_ylim(0, -5.5)

# Customize tick labels
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: format(x, '.0f') if x >= 1 else format(x, '.1f')))
ax.xaxis.set_minor_formatter(ticker.NullFormatter())
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.2))

# Customize grid
ax.grid(True, which='major', linestyle='-', alpha=0.7)
ax.grid(True, which='minor', linestyle=':', alpha=0.4)

# Add reference lines
ax.axvline(x=1, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(1.1, -0.2, '1 kpc', color='r', alpha=0.8, rotation=90, va='bottom', fontsize=10)
# ax.axvline(x=10, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
# ax.text(11, -0.2, '10 kpc', color='r', alpha=0.8, rotation=90, va='bottom', fontsize=10)
# ax.axvline(x=100, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
# ax.text(110, -0.2, '100 kpc', color='r', alpha=0.8, rotation=90, va='bottom', fontsize=10)
#vertical lines for 7.94, 12.59, 19.95, 31.62, 50.12, 79.43
ax.axvline(x=7.94, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(8.94, -0.2, '7.94 kpc', color='r', alpha=0.8, rotation=90, va='bottom', fontsize=10)
ax.axvline(x=12.59, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(13.59, -0.2, '12.59 kpc', color='r', alpha=0.8, rotation=90, va='bottom', fontsize=10)
ax.axvline(x=19.95, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(20.95, -0.2, '19.95 kpc', color='r', alpha=0.8, rotation=90, va='bottom', fontsize=10)
ax.axvline(x=31.62, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(32.62, -0.2, '31.62 kpc', color='r', alpha=0.8, rotation=90, va='bottom', fontsize=10)
ax.axvline(x=50.12, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(51.12, -0.2, '50.12 kpc', color='r', alpha=0.8, rotation=90, va='bottom', fontsize=10)
ax.axvline(x=79.43, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(80.43, -0.2, '79.43 kpc', color='r', alpha=0.8, rotation=90, va='bottom', fontsize=10)
# vertical line at 3 kpc, call it selected GAIA limit
ax.axvline(x=3, color='blue', linestyle='--', alpha=0.7, linewidth=1.5)
ax.text(3.1, -0.2, '3 kpc', color='blue', alpha=0.8, rotation=90, va='bottom', fontsize=10)


# Add annotations
ax.text(0.95, 0.05, 'Pan-STARRS g-band\nbright limit: 14.5 mag', 
        transform=ax.transAxes, fontsize=10, va='bottom', ha='right', 
        bbox=dict(boxstyle='round,pad=0.5', fc='yellow', ec='k', alpha=0.7))

plt.tight_layout()
plt.savefig('absmag_vs_distance.png', dpi=300)
plt.show()

print("Example values:")
for M in [-5, -4, -3, -2, -1, 0]:
    d = 10**((m_app - M + 5) / 5) / 1000
    print(f"Absolute Magnitude: {M:.1f}, Distance: {d:.2f} kpc")