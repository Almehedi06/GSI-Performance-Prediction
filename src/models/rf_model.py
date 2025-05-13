from sklearn.ensemble import RandomForestRegressor
import os
import joblib

def build_rf_model(n_estimators=50, max_depth=10, min_samples_leaf=4, min_samples_split=2, random_state=42):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        random_state=random_state
    )

def save_rf_model(model, path="results/models/rf_model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"\n✅ RF model saved at: {path}")
