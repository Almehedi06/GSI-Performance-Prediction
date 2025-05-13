import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import os

def build_lstm_model(input_shape, lstm_cfg):
    model = Sequential([
        LSTM(units=lstm_cfg["units_1"], return_sequences=True, input_shape=input_shape),
        Dropout(lstm_cfg["dropout_1"]),
        BatchNormalization(),
        LSTM(units=lstm_cfg["units_2"]),
        Dropout(lstm_cfg["dropout_2"]),
        Dense(1)
    ])
    optimizer = Adam(learning_rate=lstm_cfg["learning_rate"])
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def reshape_sequences(X, y, timesteps):
    n_samples = X.shape[0]
    X_seq = np.array([X[i - timesteps:i] for i in range(timesteps, n_samples)])
    y_seq = y[timesteps:].reset_index(drop=True)
    return X_seq, y_seq

def save_lstm_model(model, path="results/models/lstm_model.keras"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    model.save(path)
    print(f"\n✅ LSTM model saved at: {path}")
