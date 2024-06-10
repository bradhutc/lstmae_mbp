import pandas as pd
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u

# Load the data
stars_path = 'C:/Users/Bradl/OneDrive/SDSS-LSTMAE/LatestStars.csv'
Stars = pd.read_csv(stars_path)
Stars['g-i'] = Stars['g'] - Stars['i']

# Initial filtering based on criteria
Stars = Stars[
    (Stars[['err_u', 'err_g', 'err_r', 'err_i', 'err_z']] >= 0).all(axis=1) &
    (Stars[['err_u', 'err_g', 'err_r', 'err_i', 'err_z']] <= 0.05).all(axis=1) &
    (Stars[['u', 'g', 'r', 'i', 'z']] < 22).all(axis=1) &
    (Stars['g-i'] < 6)
]

print(f"Loaded {len(Stars)} stars from 'LatestStars.csv'.")
# Conversion from RA and dec to Galactic l and b
coords = SkyCoord(ra=Stars['ra'].values*u.degree, dec=Stars['dec'].values*u.degree, frame='icrs')
Stars['l'] = coords.galactic.l.degree
Stars['b'] = coords.galactic.b.degree

# Adjusting 'l' values if necessary
Stars['l'] = Stars['l'].apply(lambda l: l-360 if l > 5 else l)

# Save the cleaned data
Stars['id'] = range(1, len(Stars) + 1)

Stars.to_csv('C:/Users/Bradl/OneDrive/SDSS-LSTMAE/Stars_cleaned.csv', index=False)

# Plotting
plt.figure(figsize=(12, 6))
# plt.scatter(Stars['g-i'], Stars['i'], c='black', alpha=0.1, label='Stars', s=.1)
plt.hexbin(Stars['g-i'], Stars['i'], gridsize=300, cmap='viridis', mincnt=1)
plt.gca().invert_yaxis()  # Invert the y-axis to match the astronomical convention
plt.xlabel('g-i')
plt.ylabel('i')
plt.title('Color-Magnitude Diagram for SDSS Stars')
plt.savefig('Color_Magnitude.png', format='png', dpi=500)
# plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(Stars['l'], Stars['b'], c='black', alpha=0.1, label='Stars', s=1)
plt.xlabel('Galactic Longitude (l)')
plt.ylabel('Galactic Latitude (b)')
plt.title('Galactic Coordinates of SDSS Stars')
plt.savefig('Galactic_Coordinates.png', format='png', dpi=500)
# plt.show()

plt.figure(figsize=(12, 6))
plt.scatter(Stars['ra'], Stars['dec'], c='black', alpha=0.1, label='Stars', s=1)
plt.xlabel(r'$\alpha$ (\degrees)')
plt.ylabel(r'$\delta$ (\degrees)')
plt.title('RA vs Dec for SDSS Stars')
plt.savefig('RA_Dec.png', format='png', dpi=500)

print(min(Stars['ra']), max(Stars['ra'])) 
print(min(Stars['dec']), max(Stars['dec']))