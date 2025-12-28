import os
import joblib
import pandas as pd
import numpy as np
from keras.models import load_model

# Path configurations
MODEL_DIR = "/mnt/c/Users/phili/Downloads/PI/PI 2/ml_models"
MODEL_PATH = os.path.join(MODEL_DIR, "throughput_model.keras")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.gz")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_names.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "encoder.gz")

# Load components
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)
encoder = joblib.load(ENCODER_PATH)

# Sample input data (EXCLUDE TARGETS)
sample = {
    "run_num": 1,
    "seq_num": 1,
    "abstractSignalStr": "example",
    "latitude": -93.268,
    "longitude": 44.9808,
    "movingSpeed": 5.5,
    "compassDirection": 45,
    "nrStatus": 1,
    "lte_rssi": -80,
    "lte_rsrp": -90,
    "lte_rsrq": -10,
    "lte_rssnr": 20,
    "nr_ssRsrp": -85,
    "nr_ssRsrq": -10,
    "nr_ssSinr": 20,
    "mobility_mode": "Walking",
    "trajectory_direction": 45,
    "tower_id": 1,
    "delta": 5,
    "variation_relative": 10,
    "slope_brut": 1,
    "slope_lisse": 1,
    "seuil": 10,
}

# Create DataFrame
df = pd.DataFrame([sample])

# ─── Preprocessing Pipeline ────────────────────────────────────────
# 1. Encode categorical features
categorical_cols = ["abstractSignalStr", "mobility_mode"]
encoded_features = encoder.transform(df[categorical_cols])
encoded_df = pd.DataFrame(
    encoded_features, columns=encoder.get_feature_names_out(categorical_cols)
)

# 2. Merge features
df_processed = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

# 3. Filter columns to match training features
available_features = [col for col in feature_names if col in df_processed.columns]
missing_features = [col for col in feature_names if col not in df_processed.columns]

# Add missing features with 0s
for col in missing_features:
    df_processed[col] = 0

df_processed = df_processed[feature_names]

# 4. Convert to numpy array and scale
X = df_processed.to_numpy()
scaled_data = scaler.transform(X)

# ─── Sequence Handling ──────────────────────────────────────────────
sequence = np.tile(scaled_data, (10, 1)).reshape((1, 10, -1))

# ─── Prediction ─────────────────────────────────────────────────────
prediction = model.predict(sequence)
print(f"Predicted throughput (raw): {prediction[0][0][0]:.2f}")
print(f"Predicted throughput (smooth): {prediction[1][0][0]:.2f}")
# é
