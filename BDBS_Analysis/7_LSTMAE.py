import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed

# file_path='C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/Disk_BC.csv'
file_path ='C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/gaiaupload.csv'
# data = pd.read_csv(file_path)[['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']]
full_data = pd.read_csv(file_path)
print(f'Loaded {len(full_data)} stars from {file_path}.')
data = full_data[['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag']]


# scaler = StandardScaler()
# scaler = RobustScaler()
print(len(data))

data_array = data.to_numpy()

# Reshape data to [samples, timesteps, features] for LSTM
data_array_reshaped = data_array.reshape((data_array.shape[0], 1, data_array.shape[1]))

# Now, split into train and test data
train_data, test_data = train_test_split(data_array_reshaped, test_size=0.1, random_state=31)

# First, flatten the data for scaling
n_samples, n_timesteps, n_features = train_data.shape
train_data_flattened = train_data.reshape((n_samples, n_timesteps * n_features))
test_data_flattened = test_data.reshape((test_data.shape[0], n_timesteps * n_features))

# Apply scaling
scaler = MinMaxScaler()
train_data_scaled = scaler.fit_transform(train_data_flattened)
test_data_scaled = scaler.transform(test_data_flattened)

# Reshape back to [samples, timesteps, features]
train_data = train_data_scaled.reshape((n_samples, n_timesteps, n_features))
test_data = test_data_scaled.reshape((test_data.shape[0], n_timesteps, n_features))



model = Sequential()
model.add(LSTM(2, activation='elu', input_shape=(1, 6)))
model.add(RepeatVector(1))
# model.add(LSTM(8, activation='elu', return_sequences=True))
# model.add(LSTM(32, activation='elu', return_sequences=True))
model.add(LSTM(256, activation='elu', return_sequences=True))
model.add(TimeDistributed(Dense(6)))
model.compile(optimizer='adam', loss='mse')


learning_rate = 0.0001
batch_size = 5012

# Train the model
history = model.fit(train_data, train_data, epochs=2, batch_size=batch_size, validation_data=(test_data, test_data), verbose=1)

# model.summary()

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig('loss_vs_epochs.png', format='png', dpi=500)
plt.close()


# Predict and evaluate
predicted = model.predict(train_data, verbose=0)

# Reshape and inverse transform to original scale
predicted = predicted.reshape((predicted.shape[0], predicted.shape[2]))
predicted = scaler.inverse_transform(predicted)

test_loss = model.evaluate(test_data, test_data)
print(f"Test Loss: {test_loss}")

encoder = Model(inputs=model.inputs, outputs=model.layers[0].output)

# Extract latent space
latent_space_representations = encoder.predict(train_data)

# Get the reconstructed outputs
reconstructed = model.predict(train_data)

reconstructed_original_scale = scaler.inverse_transform(reconstructed.reshape(-1, 6))

# Step 1: Predict the data
reconstructed_data = model.predict(train_data)

# Step 2: Extract latent space
latent_space_values = encoder.predict(train_data)

original_train_data_inverse = scaler.inverse_transform(train_data.reshape(-1, 6))
reconstructed_data_inverse = scaler.inverse_transform(reconstructed_data.reshape(-1, 6))

matched_data = pd.read_csv('C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/simbad_matched.csv')

# Step 3: Combine the data into a DataFrame
combined_data = pd.DataFrame(original_train_data_inverse, columns=['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'])
combined_data[['u_reconstructed', 'g_reconstructed', 'r_reconstructed', 'i_reconstructed', 'z_reconstructed', 'y_reconstructed']] = reconstructed_data_inverse
combined_data[['latent_dim1', 'latent_dim2']] = latent_space_values.reshape(-1, 2)

# Step 4: Save to CSV file & Plot Results
output_file_path = 'combined_data.csv'
combined_data.to_csv(output_file_path, index=False)
output_dir = 'magnitude_reconstructions'
os.makedirs(output_dir, exist_ok=True)

merge_with_full = pd.merge(combined_data, full_data, left_on=['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'], right_on=['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'], how='inner')

merged_data = pd.merge(merge_with_full, matched_data, left_on=['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'], right_on=['umag', 'gmag', 'rmag', 'imag', 'zmag', 'ymag'], how='inner')

# Map OTYPE to numerical values
otype_mapping = {otype: i for i, otype in enumerate(merged_data['object_type'].unique())}
merged_data['OTYPE_numeric'] = merged_data['object_type'].map(otype_mapping)

palette = sns.color_palette("husl", n_colors=len(merged_data['object_type'].unique()))

plt.figure(figsize=(12, 8))

marker_styles = {'RRLyrae': 's', 'HorBranch*': 'D', 'EllipVar': '^', 'RGB*': '*', 'delSctV*': 'x', 'EclBin': 'p', 'HotSubdwarf_Candidate': '*', 'ChemPec*' : 'v', 'Radio': 'o', 'LongPeriodV*_Candidate': 'o', 'LongPeriodV*': 'o','ChemPec*': '^', 'Variable*':'*', 'PulsV': 'v'}
# Loop through each star type and plot with corresponding marker style
plt.scatter(combined_data['latent_dim1'], combined_data['latent_dim2'], c='grey', s=0.2, alpha=0.3)
for otype, marker in marker_styles.items():
    otype_data = merged_data[merged_data['object_type'] == otype]
    print(otype_data.columns)
    plt.scatter(otype_data['latent_dim1'], otype_data['latent_dim2'], label=otype, marker=marker, s=30, alpha=0.9)
plt.xlabel('CSFS x')
plt.ylabel('CSFS y')
plt.legend()
plt.savefig('CSFS.png', format='png', dpi=500)
