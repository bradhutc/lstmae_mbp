import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_mae(actuals, predictions):
    """ Calculate the mean absolute error between actuals and predictions """
    return abs(actuals - predictions).mean()

def plot_figures(data):
    """ Plot both latent dimensions and color-magnitude diagram in one figure """
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # Plotting the latent space
    scatter = axs[0].scatter( data['Gamma'],data['Lambda'], c=data['label'], s=1, cmap='Set1')
    axs[0].invert_xaxis()
    # axs[0].invert_yaxis()
    axs[0].set_title('Composite Stellar Feature Space')
    axs[0].set_ylabel(r'$\Lambda$')
    axs[0].set_xlabel(r'$\Gamma$')
    plt.colorbar(scatter, ax=axs[0], label='Isochrone Age (Dex)')
    axs[0].grid(True)

    # Calculating g-r for color-magnitude diagram
    data['g-r'] = data['g'] - data['r']


    scatter = axs[1].scatter(data['g-i'], data['i'], c=data['label'], s=1, cmap='Set1')

    axs[1].set_title('Color-Magnitude Diagram')
    axs[1].set_xlabel('g - i')
    axs[1].set_ylabel('i')
    plt.colorbar(scatter, ax=axs[1], label='Isochrone Age (Dex)')
    axs[1].invert_yaxis()
    axs[1].grid(True)
    plt.savefig('CSFS.png', dpi=300)

def main():
    # Load the data
    data = pd.read_csv('CSFSmultipleisochrones.csv')
    # Specify the bands and their corresponding prediction columns
    bands = ['u', 'g', 'r', 'i', 'z', 'J', 'H', 'Ks']
    reconstructed_bands = ['reconstructed_u', 'reconstructed_g', 'reconstructed_r', 
                           'reconstructed_i', 'reconstructed_z', 'rec_j', 'rec_h', 'rec_k']
    
    # Calculate MAE for each band and store the results
    errors = []
    for actual, predicted in zip(bands, reconstructed_bands):
        mae = calculate_mae(data[actual], data[predicted])
        errors.append(mae)
        print(f"MAE for {actual}: {round(mae,3)}")
    
    # Calculate the overall mean of these MAEs
    overall_mae = round(sum(errors) / len(errors),3)
    print(f"Overall Mean Absolute Error across all bands: {overall_mae}")

    # Prepare the data for plotting
    data['g-i'] = data['g'] - data['i']
    plot_figures(data)

if __name__ == "__main__":
    main()
