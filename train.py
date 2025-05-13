from config.config_rf import CONFIG
from src.models.rf_model import build_rf_model

# Build and train model using config parameters
model = build_rf_model(
    n_estimators=CONFIG["n_estimators"],
    max_depth=CONFIG["max_depth"],
    min_samples_leaf=CONFIG["min_samples_leaf"],
    min_samples_split=CONFIG["min_samples_split"],
    random_state=CONFIG["random_state"]
)
