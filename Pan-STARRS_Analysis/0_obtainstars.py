import pandas as pd
import matplotlib.pyplot as plt

fp='/N/project/catypGC/Bradley/PanStarrs/hlsp_ps1-strm_ps1_gpc1_p25-p28_multi_v1_cat.csv.gz'
column_names = [
    'objID', 'uniquePspsOBid', 'raMean', 'decMean', 'l', 'b', 'class',
    'prob_Galaxy', 'prob_Star', 'prob_QSO', 'extrapolation_Class',
    'cellDistance_Class', 'cellID_Class', 'z_phot', 'z_photErr', 'z_phot0',
    'extrapolation_Photoz', 'cellDistance_Photoz', 'cellID_Photoz'
]
batch_size = 10000  # Adjust the batch size as per your requirements
batches = pd.read_csv(fp, compression='gzip', chunksize=batch_size, names=column_names)
data_observed = 0
stars_found = 0
# Empty dataframe that will store the probable stars.
star_df = pd.DataFrame(columns=column_names)
# Process each batch
for batch in batches:
    # print(batch.head())
    # print(batch.columns)
    like_star = batch[batch['prob_Star'] > 0.9]
    star_df = pd.concat([star_df, like_star], ignore_index=True)
    data_observed += batch_size
    stars_found += len(like_star)
    print(f'{len(like_star)} Stars Found for Batch of {batch_size},{stars_found} Stars Found Total Out of {data_observed} Sources Observed')

print(f'{len(star_df)} Stars found in PanStarrs')
print(star_df.head())
star_df.to_csv('stars_from_panstarrs_nophot.csv', index=False)
print(f'Dataframe Saved')

plt.figure(figsize=(10, 8))
plt.hexbin(star_df['l'], star_df['b'], gridsize=150, cmap='viridis')
plt.xlabel('Galactic Longitude (l)')
plt.ylabel('Galactic Latitude (b)')
plt.title('Hexbin Plot of Galactic Latitude vs Galactic Longitude for Stars from PanStarrs')
plt.colorbar(label='Count')
plt.tight_layout()
plt.savefig('l_b_plot.png')
