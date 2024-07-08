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
        # self.model.add(TimeDistributed(Dense(5)))
        # self.model.compile(optimizer='adam', loss='mse')
        # self.encoder = Model(inputs=self.model.inputs, outputs=self.model.layers[0].output)
        self.model = Sequential()
        self.model.add(LSTM(128, activation='elu', input_shape=(input_shape, 6), return_sequences=True))
        # self.model.add(LSTM(64, activation='elu', return_sequences=True))
        # self.model.add(LSTM(32, activation='elu', return_sequences=True))
        self.model.add(LSTM(2, activation='elu', return_sequences=False))
        self.model.add(RepeatVector(input_shape))
        # self.model.add(LSTM(32, activation='elu', return_sequences=True))
        # self.model.add(Dropout(0.2))
        # self.model.add(LSTM(64, activation='elu', return_sequences=True))
        self.model.add(LSTM(128, activation='elu', return_sequences=True))
        self.model.add(LSTM(6, activation='elu', return_sequences=True))
    
        self.model.compile(optimizer='Adam', loss='mse')
        self.encoder = Model(inputs=self.model.inputs, outputs=self.model.layers[1].output)

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
    reconstructed_magnitudes = autoencoder.reconstruct(data).reshape(-1, 6)

    # Prepare the DataFrame
    original_data_scaled_back = scaler.inverse_transform(data.reshape(-1, 6))
    df = pd.DataFrame(original_data_scaled_back, columns=['psfMag_u','gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag'])
    df[['Gamma', 'Lambda']] = latent_representations
    df[['psfMag_u_prime','gMeanPSFMag_prime', 'rMeanPSFMag_prime', 'iMeanPSFMag_prime', 'zMeanPSFMag_prime', 'yMeanPSFMag_prime']] = scaler.inverse_transform(reconstructed_magnitudes)
    
    # Calculate R-squared score for each magnitude
    r2_scores = {}
    for mag in ['psfMag_u','gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag']:
        r2_scores[f'{mag}_r2'] = r2_score(df[mag], df[f'{mag}_prime'])
    
    df = pd.concat([df, pd.DataFrame(r2_scores, index=df.index)], axis=1)
    df['dataset_type'] = dataset_type
    df['raMean'] = y_data['raMean'].values
    df['decMean'] = y_data['decMean'].values
    df['l'] = y_data['l'].values
    df['b'] = y_data['b'].values
    df['objID'] = y_data['objID'].values
    
    return df


file_path = 'C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/data/PS_clean.csv'
full_data = pd.read_csv(file_path)[['psfMag_u', 'gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag', 'yMeanPSFMag','objID','raMean','decMean','l','b']]
full_data = full_data.dropna()
scaler = MinMaxScaler()
data = full_data[['psfMag_u', 'gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag','yMeanPSFMag','objID','raMean','decMean','l','b']]
data_array = data.to_numpy()
features= ['psfMag_u','gMeanPSFMag', 'rMeanPSFMag', 'iMeanPSFMag', 'zMeanPSFMag','yMeanPSFMag']
X = data[features]
y = data.drop(columns=features)

# Split data into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

train_data = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
val_data = X_val_scaled.reshape((X_val_scaled.shape[0], 1, X_val_scaled.shape[1]))
test_data = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
autoencoder = LSTMAutoencoder(input_shape=1)
plotter = CompositeStellarFeatureSpace(encoder=autoencoder.encoder, data=train_data, scaler=scaler, filepath='C:/Users/Bradl/OneDrive/Astrophysics_Research/lstmae_mbp/Pan-STARRS_Analysis/plots/CSFS/')
early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)

# Now train the model with the callbacks
history = autoencoder.train(train_data, val_data, epochs=50, batch_size=512, callbacks=[plotter, early_stopping])

test_loss = autoencoder.model.evaluate(test_data, test_data)
print(f'Test Loss: {test_loss}')

# Prepare and combine data frames with correct labels
train_df = prepare_data(autoencoder, train_data, scaler, 'train', y_train)
val_df = prepare_data(autoencoder, val_data, scaler, 'validation', y_val)
test_df = prepare_data(autoencoder, test_data, scaler, 'test', y_test)
complete_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
# Save the DataFrame to CSV
complete_df.to_csv('CSFS_ugrizy.csv', index=False)

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
