import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed, Dropout
from keras.callbacks import Callback, EarlyStopping
from keras.optimizers import Adam
import gzip
import numpy as np


class LSTMAutoencoder:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(2, activation='elu', input_shape=(input_shape, 8)))
        self.model.add(RepeatVector(1))
        self.model.add(TimeDistributed(Dense(8)))
        self.model.compile(optimizer='adam', loss='mse')
        self.encoder = Model(inputs=self.model.inputs, outputs=self.model.layers[0].output)
        
        
    def train(self, train_data, test_data, epochs, batch_size, callbacks=None):
        history = self.model.fit(train_data, train_data, epochs=epochs, batch_size=batch_size,
                                 validation_data=(test_data, test_data), verbose=1, callbacks=callbacks)
        return history

    def predict(self, data):
        return self.model.predict(data)

    def extract_latent_space(self, data):
        if self.encoder:
            return self.encoder.predict(data)
        else:
            raise ValueError("Model must be trained before extracting latent space.")
    
    def reconstruct(self, data):
        return self.model.predict(data)

class CompositeStellarFeatureSpace(Callback):
    def __init__(self, encoder, data, scaler, filepath):
        super().__init__()
        self.encoder = encoder
        self.data = data
        self.scaler = scaler
        self.filepath = filepath

    def on_epoch_end(self, epoch, logs=None):
        latent_space = self.encoder.predict(self.data)
        plt.scatter(latent_space[:, 0], latent_space[:, 1], c='black', s=1, alpha=0.5)
        # plt.gca().invert_yaxis()
        plt.gca().invert_xaxis()
        plt.xlabel(r'$\Gamma$')
        plt.ylabel(r'$\Lambda$')
        plt.title(f'Composite Stellar Feature Space after Epoch {epoch + 1}')
        file_to_save = os.path.join(self.filepath, f'CSFS_epoch_{epoch + 1}.png')
        plt.savefig(file_to_save, dpi=500)
        plt.close()

def prepare_data(autoencoder, data, scaler, dataset_type, labels):
    # Extract latent representations
    latent_representations = autoencoder.extract_latent_space(data)

    reconstructed_magnitudes = autoencoder.reconstruct(data).reshape(-1, 8)

    # Prepare the DataFrame
    original_data_scaled_back = scaler.inverse_transform(data.reshape(-1, 8))
    df = pd.DataFrame(original_data_scaled_back, columns=['u', 'g', 'r', 'i', 'z', 'J', 'H', 'Ks'])
    df[['Gamma', 'Lambda']] = latent_representations
    df[['reconstructed_u', 'reconstructed_g', 'reconstructed_r', 'reconstructed_i', 'reconstructed_z', 'rec_j', 'rec_h', 'rec_k']] = scaler.inverse_transform(reconstructed_magnitudes)
    df = pd.concat([df, labels.reset_index(drop=True)], axis=1)
    df['dataset_type'] = dataset_type

    return df

# Path to the directory containing the isochrone files
directory_path = 'C:/Users/Bradl/OneDrive/Astrophysics_Research/Isochrone_Analysis/Isochrones/'
csfs_directory = os.path.join(directory_path, 'CSFS')

# Ensure the directory exists
os.makedirs(csfs_directory, exist_ok=True)

# Initialize an empty list to store the DataFrames
data_frames = []

column_names = [
    'Z', 'age', 'Mini', 'Mass', 'logL', 'logTe', 'logg', 'label', 'McoreTP', 'C_O', 
    'period0', 'period1', 'period2', 'period3', 'period4', 'pmode', 'Mloss', 'tau1m', 
    'X', 'Y', 'Xc', 'Xn', 'Xo', 'Cexcess', 'Zextra', 'mbolmag', 'umag', 'gmag', 
    'rmag', 'imag', 'zmag', 'Jmag', 'Hmag', 'Ksmag'
]

for age in np.arange(7.0, 10.5, 0.4):
    file_name = f'{age:.1f}.dat'
    file_path = os.path.join(directory_path, file_name)
    if os.path.exists(file_path):
        temp_df = pd.read_csv(file_path, delimiter='\s+', names=column_names)
        temp_df['label'] = f'{age:.1f}'
        data_frames.append(temp_df)
    else:
        print(f"File {file_name} not found. Skipping...")

features = ['umag', 'gmag', 'rmag', 'imag', 'zmag', 'Jmag', 'Hmag', 'Ksmag']

data = pd.concat(data_frames, ignore_index=True)
print(data.head())
X = data[features]
y = data.drop(columns=features)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=43)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

train_data = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
test_data = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

autoencoder = LSTMAutoencoder(input_shape=1)
plotter = CompositeStellarFeatureSpace(encoder=autoencoder.encoder, data=train_data, scaler=scaler, filepath=csfs_directory)
early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min', restore_best_weights=True)

# Now train the model with the callbacks
history = autoencoder.train(train_data, test_data, epochs=15, batch_size=512, callbacks=[plotter, early_stopping])

# Prepare and combine data frames with correct labels
train_df = prepare_data(autoencoder, train_data, scaler, 'train', y_train.reset_index(drop=True))
test_df = prepare_data(autoencoder, test_data, scaler, 'test', y_test.reset_index(drop=True))
complete_df = pd.concat([train_df, test_df], ignore_index=True)

# Save the DataFrame to CSV
complete_df.to_csv('CSFSmultipleisochrones.csv', index=False)

# Save loss and validation loss to CSV
history_df = pd.DataFrame(history.history)
history_df.to_csv('training_history.csv', index=False)

plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Model Loss')
plt.ylabel('Log Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig('log_loss_vs_epochs.png', format='png', dpi=500)
plt.close()
