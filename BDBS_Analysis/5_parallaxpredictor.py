import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer, PowerTransformer
from keras.layers import Dense, Dropout, LeakyReLU, ReLU
from keras.regularizers import l2
import matplotlib.pyplot as plt
from keras.callbacks import Callback
import os
import numpy as np

class ParallaxPredictor(keras.Model):
    def __init__(self):
        super(ParallaxPredictor, self).__init__()
        self.dense1 = Dense(256, activation=LeakyReLU(alpha=0.1), kernel_regularizer=l2(0.001))
        self.dropout1 = Dropout(0.001)
        self.dense2 = Dense(128, activation=LeakyReLU(alpha=0.1), kernel_regularizer=l2(0.001))
        self.dropout2 = Dropout(0.001)
        self.output_layer = Dense(1, activation='linear')
        
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return self.output_layer(x)

class PlotParallaxMassesCallback(Callback):
    def __init__(self, X_test, Y_test, model, scaler_Y, epoch_interval=10, output_dir='plots'):
        self.X_test = X_test
        self.Y_test = Y_test
        self.model = model
        self.scaler_Y = scaler_Y
        self.epoch_interval = epoch_interval
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        predicted_scaled = self.model.predict(self.X_test)
        predicted = self.scaler_Y.inverse_transform(predicted_scaled)
        original = self.scaler_Y.inverse_transform(self.Y_test)

        all_values = np.concatenate([predicted.flatten(), original.flatten()])
        bins = np.linspace(all_values.min(), all_values.max(), 100)
        mean_pred_pi, std_pred_pi = np.mean(predicted[:, 0]), np.std(predicted[:, 0])
        mean_orig_pi, std_orig_pi = np.mean(original[:, 0]), np.std(original[:, 0])
        plt.figure(figsize=(10, 6))
        plt.hist(predicted[:, 0], bins=bins, alpha=0.5,color = 'maroon', label=fr'Predicted $\pi$ (mean: {mean_pred_pi:.2f}, std: {std_pred_pi:.2f})', density=True)
        plt.hist(original[:, 0], bins=bins, alpha=0.5,color='cadetblue', label=fr'Original $\pi$ (mean: {mean_orig_pi:.2f}, std: {std_orig_pi:.2f})', density=True)
        plt.xlabel(r'$\pi$ (mas)')
        plt.ylabel('Frequency')
        plt.title(f'Parallax Prediction from Position, Colors, and Magnitudes (Epoch {epoch + 1})')
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, f'epoch_{epoch + 1}.png'))
        plt.close()

if __name__ == '__main__':
    df = pd.read_csv('C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/Cleaned_Gaia_Results.csv')
    print(df.columns)
    print(df.head())
    data = df[(df['parallax_error'] < 0.05) & (df['parallax'] < 1.0)]
    print(f'Number of stars: {len(data)}')

    X, Y = data.drop(columns=['gaia_id','ra','dec', 'BDBS_ID', 'pmra','pmdec','parallax', 'parallax_error','pmra_error','pmdec_error','U','G','R','I','Z','Y', 'd_kpc'], axis=1), data[['parallax']]
   
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    scaler_X = RobustScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    scaler_Y = RobustScaler()
    Y_train_scaled = scaler_Y.fit_transform(Y_train)
    Y_test_scaled = scaler_Y.transform(Y_test)
    
    
    batch_size = 256
    
    model = ParallaxPredictor()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mse'])
    plot_callback = PlotParallaxMassesCallback(X_test_scaled, Y_test_scaled, model, scaler_Y, epoch_interval=5)

    history = model.fit(
        X_train_scaled, Y_train_scaled, 
        batch_size=batch_size, epochs=115,
        validation_split=0.2, callbacks=[plot_callback]
        )
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig('loss.png')
    plt.show()

    test_loss, test_mse = model.evaluate(X_test_scaled, Y_test_scaled)
    print(f"Test Loss: {test_loss}, Test MAE: {test_mse}")
    # Predictions
    predictions_scaled = model.predict(X_test_scaled)

    # Inverse transform predictions to original scale
    predictions = scaler_Y.inverse_transform(predictions_scaled)

    print(predictions)
    
    test_results_df = pd.DataFrame(X_test, columns=X_test.columns)
    test_results_df['Actual Parallax'] = Y_test.values.flatten()
    test_results_df['Predicted Parallax'] = predictions.flatten()

  
    test_results_csv_path = 'test_results.csv'
    test_results_df.to_csv(test_results_csv_path, index=False)
    print(f"Saved the test results to {test_results_csv_path}")

# -------------------------------------------------------------------

    all_stars = pd.read_csv('C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/Cleaned_BDBS_Results.csv')

    all_stars_features = all_stars.drop(columns=['gaia_id', 'BDBS_ID', 'ra','dec'])
    all_stars_scaled = scaler_X.transform(all_stars_features)
    predicted_outputs = model.predict(all_stars_scaled)
    predicted_outputs_scaled = scaler_Y.inverse_transform(predicted_outputs)
    # Extract predictions for each output
    predicted_parallax = predicted_outputs.flatten() 
    all_stars['Predicted Parallax'] = predicted_parallax

    all_stars.to_csv('C:/Users/Bradl/OneDrive/BDBS-bradhutc-pc/BDBS_W_Parallax.csv', index=False)