import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.models import Sequential
from keras.layers import LSTM, Dense, RepeatVector, TimeDistributed
from keras.callbacks import Callback

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
        plt.savefig(f'{self.filepath}_epoch_{epoch+1}.png', format='png', dpi=500)
        plt.close()

def prepare_data(autoencoder, data, scaler, dataset_type):
    # Extract latent representations
    latent_representations = autoencoder.extract_latent_space(data)

    # Reconstruct the magnitudes
    reconstructed_magnitudes = autoencoder.reconstruct(data).reshape(-1, 5)

    # Prepare the DataFrame
    original_data_scaled_back = scaler.inverse_transform(data.reshape(-1, 5))
    df = pd.DataFrame(original_data_scaled_back, columns=['u', 'g', 'r', 'i', 'z'])
    df[['Gamma', 'Lambda']] = latent_representations
    df[['u_prime', 'g_prime', 'r_prime', 'i_prime', 'z_prime']] = scaler.inverse_transform(reconstructed_magnitudes)
    df['dataset_type'] = dataset_type

    return df


file_path = 'C:/Users/Bradl/OneDrive/SDSS-LSTMAE/cleaned_stellar_data.csv'
full_data = pd.read_csv(file_path)
scaler = MinMaxScaler()
data = full_data[['u', 'g', 'r', 'i', 'z']]
data_array = data.to_numpy()
# data_array_reshaped = data_array.reshape((data_array.shape[0], 1, data_array.shape[1]))
train_data, test_data = train_test_split(data_array, test_size=0.2, random_state=31)
train_data= scaler.fit_transform(train_data)
test_data = scaler.transform(test_data)
train_data = train_data.reshape((train_data.shape[0], 1, train_data.shape[1]))
test_data = test_data.reshape((test_data.shape[0], 1, test_data.shape[1]))


autoencoder = LSTMAutoencoder(input_shape=1)
plotter = CompositeStellarFeatureSpace(encoder=autoencoder.encoder, data=train_data, scaler=scaler, filepath='C:/Users/Bradl/OneDrive/SDSS-LSTMAE/plots')
history = autoencoder.train(train_data, test_data, epochs=3, batch_size=16, callbacks=[plotter])

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


train_df = prepare_data(autoencoder, train_data, scaler, 'train')
test_df = prepare_data(autoencoder, test_data, scaler, 'test')

# Combine both DataFrames
complete_df = pd.concat([train_df, test_df], ignore_index=True)

# Save to CSV
# complete_df.to_csv('CSFS.csv', index=False)