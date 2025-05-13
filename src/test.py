import argparse
import yaml
import joblib
from src.utils.data_loader import load_and_preprocess_data
from src.utils.evaluation import evaluate_model

# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, help="Model name to override config.yaml (e.g., rf, lstm)")
args = parser.parse_args()

# Load YAML config
with open("config/config.yaml", "r") as file:
    config = yaml.safe_load(file)

# Determine model name from CLI or config
model_name = args.model if args.model else config["model_name"]

# Load Test Data
data_cfg = config["data"]
_, _, X_test, y_test = load_and_preprocess_data(
    data_cfg["path"],
    data_cfg["input_columns"],
    data_cfg["target_column"],
    set(data_cfg["test_storm_ids"])
)

# Load Trained Model
model_path = f"results/models/{model_name}_model.joblib"
model = joblib.load(model_path)

# Evaluate on Test Set Only
y_pred_test = model.predict(X_test)
evaluate_model(y_test, y_pred_test, f"Test Only - {model_name.upper()}")
