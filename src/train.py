import argparse
import yaml
import os
from src.utils.data_loader import load_and_preprocess_data
from src.utils.evaluation import evaluate_model
from src.models.rf_model import build_rf_model, save_rf_model

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="Model name to override config.yaml (e.g., rf, lstm)")
args = parser.parse_args()

# Load YAML config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Determine model name from CLI or config
model_name = args.model if args.model else config["model_name"]

# Load Data
data_cfg = config["data"]
X_train, y_train, X_test, y_test = load_and_preprocess_data(
    data_cfg["path"],
    data_cfg["input_columns"],
    data_cfg["target_column"],
    set(data_cfg["test_storm_ids"])
)

# Model Selection
if model_name == "rf":
    rf_cfg = config["rf"]
    model = build_rf_model(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        min_samples_split=rf_cfg["min_samples_split"],
        random_state=rf_cfg["random_state"]
    )
else:
    raise ValueError(f"Model '{model_name}' is not supported yet.")

# Train and Evaluate
model.fit(X_train, y_train)
evaluate_model(y_train, model.predict(X_train), "Train")
evaluate_model(y_test, model.predict(X_test), "Test")

# Save the model
if model_name == "rf":
    save_rf_model(model)
