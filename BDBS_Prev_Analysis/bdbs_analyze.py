import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture

def plot_parallax(df):
    plt.figure(figsize=(10, 8))
    plt.scatter(df['parallax_error'], df['parallax'], c='cadetblue', s=0.5, alpha=0.2, label='Disk Stars')

    # Convert distances to parallaxes for bulge boundaries (distances in kpc, converted to milliarcseconds)
    center_of_bulge_parallax = 1 / 8.178 
    start_of_bulge_parallax = 1 / (8.178-1.75)  
    end_of_bulge_parallax = 1 / (8.178+1.75) 

    start_of_long_bar = 1 / (8.178-2.25)  # 11.5 kpc
    end_of_long_bar = 1 / (8.178+2.25)  # 5.5 kpc
    
    plt.axhline(y=center_of_bulge_parallax, color='r', linestyle='-', label='Center of Bulge')
    plt.axhline(y=start_of_bulge_parallax, color='g', linestyle='-', label='Near Edge of Bar')
    plt.axhline(y=end_of_bulge_parallax, color='g', linestyle='-', label='Far Edge of Bar')
    # plt.axhline(y=start_of_long_bar, color='b', linestyle='-', label='Near Edge of Long Bar')
    # plt.axhline(y=end_of_long_bar, color='b', linestyle='-', label='Far Edge of Long Bar')
    plt.fill_betweenx([start_of_bulge_parallax, end_of_bulge_parallax], 0, 0.4, color='green', alpha=0.2, label='Bulge')
    # Identifying possible bulge stars including parallax error consideration
    possible_bulge_lower = df['parallax'] - df['parallax_error']
    possible_bulge_upper = df['parallax'] + df['parallax_error']
    
    # Considering stars as possible bulge candidates if their parallax +/- error overlaps with the bulge range
    possible_bulge = df[(possible_bulge_lower <= start_of_bulge_parallax) & (possible_bulge_upper >= end_of_bulge_parallax)]
    
    # Highlighting these stars
    plt.scatter(possible_bulge['parallax_error'], possible_bulge['parallax'], c='maroon', s=1, alpha=0.1, label='Possible Bulge Stars')

    plt.xlabel(r'$\sigma_{\pi}$ (milliarcseconds)')
    plt.ylabel(r'$\pi$ (milliarcseconds)')
    plt.xlim(0, 0.18)
    plt.ylim(0, 0.31)
    plt.title('Parallax vs Parallax Error for BDBS Stars Cross-Matched with Gaia DR3')
    plt.legend()
    plt.show()

def fit_gmm(df):
    # Filter parallax values
    actual_parallax_filtered = df['Actual Parallax'].values.reshape(-1, 1)
    predicted_parallax_filtered = df['Predicted Parallax'].values.reshape(-1, 1)
    
    # Fit GMM
    gmm_actual = GaussianMixture(n_components=2, random_state=42).fit(actual_parallax_filtered)
    gmm_predicted = GaussianMixture(n_components=2, random_state=42).fit(predicted_parallax_filtered)
    
    # Convert means from parallax to distance for interpretation
    actual_distances_means = 1 / gmm_actual.means_.flatten()
    predicted_distances_means = 1 / gmm_predicted.means_.flatten()

    print(f'Actual Parallax GMM Parameters:\nMeans: {gmm_actual.means_.flatten()}\nConverted to Distance (kpc): {actual_distances_means}')
    print(f'Covariances: {gmm_actual.covariances_.flatten()}')
    print(f'Weights: {gmm_actual.weights_.flatten()}')
    print(f'Predicted Parallax GMM Parameters:\nMeans: {gmm_predicted.means_.flatten()}\nConverted to Distance (kpc): {predicted_distances_means}')
    print(f'Covariances: {gmm_predicted.covariances_.flatten()}')
    print(f'Weights: {gmm_predicted.weights_.flatten()}')

# Assuming you have already read your DataFrame as 'stars'
if __name__ == '__main__':
    Gaia_Stars = pd.read_csv('Cleaned_Gaia_Results.csv')
    Gaia_Stars = Gaia_Stars[(Gaia_Stars['parallax'] > 0) & (Gaia_Stars['parallax_error'] > 0)]
    # plot_parallax(Gaia_Stars)
    
    Test_Stars = pd.read_csv('C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/test_results.csv')
    fit_gmm(Test_Stars)
