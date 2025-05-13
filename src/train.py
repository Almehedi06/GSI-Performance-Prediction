import argparse
import yaml
import os
from src.utils.data_loader import load_and_preprocess_data
from src.utils.evaluation import evaluate_model
from src.models.rf_model import build_rf_model, save_rf_model
from src.models.lstm_model import build_lstm_model, reshape_sequences, save_lstm_model
from tensorflow.keras.callbacks import EarlyStopping

# ------------------------- #
# Argument Parsing First
# ------------------------- #
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="Model name to override config.yaml (e.g., rf, lstm, pilstm)")
args = parser.parse_args()

# ------------------------- #
# Load YAML Config
# ------------------------- #
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# ------------------------- #
# Resolve Model Name
# ------------------------- #
model_name = args.model if args.model else config["model_name"]
print(f"✅ Using model: {model_name}")

# ------------------------- #
# Load Data
# ------------------------- #
data_cfg = config["data"]
X_train, y_train, X_test, y_test = load_and_preprocess_data(
    data_cfg["path"],
    data_cfg["input_columns"],
    data_cfg["target_column"],
    set(data_cfg["test_storm_ids"])
)

# ------------------------- #
# Model Selection
# ------------------------- #
if model_name == "rf":
    rf_cfg = config["rf"]
    model = build_rf_model(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        min_samples_split=rf_cfg["min_samples_split"],
        random_state=rf_cfg["random_state"]
    )
    model.fit(X_train, y_train)
    evaluate_model(y_train, model.predict(X_train), "Train")
    evaluate_model(y_test, model.predict(X_test), "Test")
    save_rf_model(model)

elif model_name == "lstm":
    lstm_cfg = config["lstm"]
    timesteps = lstm_cfg["timesteps"]

    # Reshape data for LSTM
    X_train_seq, y_train_seq = reshape_sequences(X_train, y_train, timesteps)
    X_test_seq, y_test_seq = reshape_sequences(X_test, y_test, timesteps)

    input_shape = (timesteps, X_train.shape[1])
    model = build_lstm_model(input_shape, lstm_cfg)

    early_stopping = EarlyStopping(monitor='val_loss', patience=lstm_cfg["patience"], restore_best_weights=True)
    model.fit(
        X_train_seq, y_train_seq,
        epochs=lstm_cfg["epochs"],
        batch_size=lstm_cfg["batch_size"],
        validation_data=(X_test_seq, y_test_seq),
        callbacks=[early_stopping],
        verbose=1
    )

    evaluate_model(y_train_seq, model.predict(X_train_seq), "Train")
    evaluate_model(y_test_seq, model.predict(X_test_seq), "Test")
    save_lstm_model(model)

else:
    raise ValueError(f"Model '{model_name}' is not supported yet.")
