import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('CSFS.csv')

df_scaled = StandardScaler().fit_transform(df)

kmeans = KMeans(n_clusters=100, random_state=0).fit(df_scaled)

plt.figure(figsize=(12, 8))
plt.scatter(df['latent_dim1'], df['latent_dim2'], c=kmeans.labels_, cmap='viridis', s=30)
plt.xlabel('CSFS x')
plt.ylabel('CSFS y')
plt.title('CSFS Latent Space Clustering')
plt.tight_layout()
plt.savefig('CSFS_clusters.png', format='png', dpi=500)
