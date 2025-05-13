import argparse
import yaml
import joblib
import tensorflow as tf
from src.utils.data_loader import load_and_preprocess_data
from src.utils.evaluation import evaluate_model
from src.models.lstm_model import reshape_sequences

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
# Load Test Data
# ------------------------- #
data_cfg = config["data"]
_, _, X_test, y_test = load_and_preprocess_data(
    data_cfg["path"],
    data_cfg["input_columns"],
    data_cfg["target_column"],
    set(data_cfg["test_storm_ids"])
)

# ------------------------- #
# Model Loading & Evaluation
# ------------------------- #
if model_name == "rf":
    model_path = "results/models/rf_model.joblib"
    model = joblib.load(model_path)
    y_pred_test = model.predict(X_test)
    evaluate_model(y_test, y_pred_test, "Test Only - RF")

elif model_name == "lstm":
    lstm_cfg = config["lstm"]
    timesteps = lstm_cfg["timesteps"]

    # Reshape data for LSTM
    X_test_seq, y_test_seq = reshape_sequences(X_test, y_test, timesteps)

    model_path = "results/models/lstm_model.keras"
    model = tf.keras.models.load_model(model_path)
    y_pred_test = model.predict(X_test_seq)
    evaluate_model(y_test_seq, y_pred_test, "Test Only - LSTM")

else:
    raise ValueError(f"Model '{model_name}' is not supported yet.")
