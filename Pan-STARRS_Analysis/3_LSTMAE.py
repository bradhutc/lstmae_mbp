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
from sklearn.metrics import r2_score

class LSTMAutoencoder:
    def __init__(self, input_shape):
        self.model = Sequential()
        self.model.add(LSTM(2, activation='elu', input_shape=(input_shape, 5)))
        self.model.add(RepeatVector(1))
        self.model.add(TimeDistributed(Dense(5)))
        self.model.compile(optimizer='adam', loss='mse')
        self.encoder = Model(inputs=self.model.inputs, outputs=self.model.layers[0].output)

    def train(self, train_data, test_data, epochs=5, batch_size=32, callbacks=None):
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
        latent_space_representations = self.encoder.predict(self.data)
        latent_dim1, latent_dim2 = latent_space_representations[:, 0], latent_space_representations[:, 1]
        plt.figure(figsize=(12, 8))
        plt.scatter(latent_dim1, latent_dim2, c='black', s=0.2, alpha=0.3)
        plt.xlabel('Latent Dim 1')
        plt.ylabel('Latent Dim 2')
        plt.title(f'Latent Space after Epoch {epoch+1}')
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        plt.savefig(f'{self.filepath}_epoch_{epoch+1}.png', format='png', dpi=500)
        plt.close()
        
         
def prepare_data(autoencoder, data, scaler, dataset_type, y_data):
    # Extract latent representations
    latent_representations = autoencoder.extract_latent_space(data)

    # Reconstruct the magnitudes
    reconstructed_magnitudes = autoencoder.reconstruct(data).reshape(-1, 5)

    # Prepare the DataFrame
    original_data_scaled_back = scaler.inverse_transform(data.reshape(-1, 5))
    df = pd.DataFrame(original_data_scaled_back, columns=['gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag'])
    df[['Gamma', 'Lambda']] = latent_representations
    df[['gMeanPSFMag_prime', 'rMeanPSFMag_prime', 'iMeanPSFMag_prime', 'zMeanPSFMag_prime', 'yMeanPSFMag_prime']] = scaler.inverse_transform(reconstructed_magnitudes)
    
    # Calculate R-squared score for each magnitude
    r2_scores = {}
    for mag in ['gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag']:
        r2_scores[f'{mag}_r2'] = r2_score(df[mag], df[f'{mag}_prime'])
    
    df = pd.concat([df, pd.DataFrame(r2_scores, index=df.index)], axis=1)
    df['dataset_type'] = dataset_type
    
    # Include the 'objID' column from the y_data
    df['objID'] = y_data['objID'].values
    
    return df


file_path = 'C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/Pan-Starrs_Data/PS_clean.csv'
full_data = pd.read_csv(file_path)
scaler = MinMaxScaler()
data = full_data[[ 'gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag','yMeanPSFMag','objID','l','b']]
data_array = data.to_numpy()
features= ['gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag','yMeanPSFMag']
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
plotter = CompositeStellarFeatureSpace(encoder=autoencoder.encoder, data=train_data, scaler=scaler, filepath='C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/CSFS/')
early_stopping = EarlyStopping(monitor='val_loss', patience=2, mode='min', restore_best_weights=True)

# Now train the model with the callbacks
history = autoencoder.train(train_data, test_data, epochs=15, batch_size=512, callbacks=[plotter, early_stopping])

# Prepare and combine data frames with correct labels
train_df = prepare_data(autoencoder, train_data, scaler, 'train', y_train)
test_df = prepare_data(autoencoder, test_data, scaler, 'test', y_test)
complete_df = pd.concat([train_df, test_df], ignore_index=True)
# Save the DataFrame to CSV
complete_df.to_csv('CSFS_PS_Clean.csv', index=False)


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
