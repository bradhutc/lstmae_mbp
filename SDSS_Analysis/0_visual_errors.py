import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# NN_data = pd.read_csv('CSFS.csv')
# matched_data = pd.read_csv('simbad_matched.csv')
# df = pd.read_csv('Merge.csv')
# df = df[df['class'] == 0]
# magnitudes = ['u', 'g', 'r', 'i', 'z']

# for mag in magnitudes:
#     NN_data[mag] = NN_data[mag].round(2)
#     matched_data[mag] = matched_data[mag].round(2)

# merged_data = pd.merge(NN_data, matched_data, left_on=['u', 'g', 'r', 'i', 'z'], right_on=['u', 'g', 'r', 'i', 'z'], how='inner')


df = pd.read_csv('Stars_cleaned.csv')
df['g-i'] = df['g'] - df['i']
# df=df[(df['err_u'] >= 0) & (df['err_g'] >= 0) & (df['err_u'] >= 0) & (df['err_i'] >= 0) & (df['err_r'] >= 0) & (df['err_z'] >= 0) & (df['err_g'] <= 5) & (df['err_r'] <= 5) & (df['err_i'] <= 5) & (df['err_z'] <= 5) & (df['err_u'] <= 5)]
print(f"Loaded {len(df)} stars from 'Skyserver_Radial3_26_2024 11 27 48 PM.csv'.")

def plot_ra_dec(df):
    plt.figure(figsize=(12, 8))
    plt.scatter(df['ra'], df['dec'], c='black', s=5, alpha=0.3)
    plt.xlabel(r'$\alpha (\degree)$')
    plt.ylabel(r'$\delta (\degree)$')
    plt.title(f'Right Ascension vs Declination for {len(df)} SDSS Stars')
    plt.savefig('RA_Dec.png', format='png', dpi=500)
    

plot_ra_dec(df)

def plot_color_magnitude(df):
    plt.figure(figsize=(12, 8))
    plt.hexbin(df['g-i'], df['i'], gridsize=300, cmap='viridis', mincnt=1)
    plt.gca().invert_yaxis()
    plt.xlabel('g-i')
    plt.ylabel('i')
    plt.title('Color-Magnitude Diagram for SDSS Stars')
    plt.savefig('Color_Magnitude.png', format='png', dpi=500)

plot_color_magnitude(df)

def plot_mag_vs_mag_err(df,mag1, mag2):
    plt.figure(figsize=(12, 8))
    plt.scatter(df[mag1], df[mag2], c='black', s=5, alpha=0.3)
    plt.xlabel(f'{mag1}')
    plt.ylabel(f'{mag2}')
    plt.title(f'{mag1} vs {mag2} for SDSS Stars')
    plt.tight_layout()
    plt.grid()
    plt.savefig(f'{mag1}_vs_{mag2}.png', format='png', dpi=500)

plot_mag_vs_mag_err(df, 'u', 'err_u')
plot_mag_vs_mag_err(df, 'g', 'err_g')
plot_mag_vs_mag_err(df, 'r', 'err_r')
plot_mag_vs_mag_err(df, 'i', 'err_i')
plot_mag_vs_mag_err(df, 'z', 'err_z')
