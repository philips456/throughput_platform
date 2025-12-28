#!/usr/bin/env python3
"""
Script to create dummy ML model files for testing the Django application.
Run this script to generate the necessary files in the ml_models directory.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import sys

try:
    from tensorflow import keras

    KERAS_AVAILABLE = True
except ImportError:
    try:
        import keras as keras

        KERAS_AVAILABLE = True
    except ImportError:
        print("TensorFlow/Keras not available. Will create joblib model instead.")
        KERAS_AVAILABLE = False

# Create ml_models directory
model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models")
os.makedirs(model_dir, exist_ok=True)
print(f"Creating dummy ML models in: {model_dir}")

# Create dummy feature names
feature_names = [
    "longitude",
    "latitude",
    "speed",
    "direction",
    "nr_ssRsrp",
    "nr_ssRsrq",
    "nr_ssSinr",
    "lte_rsrp",
    "lte_rsrq",
    "lte_rssi",
    "lte_rssnr",
    "movingSpeed",
    "compassDirection",
    "trajectory_direction",
    "tower_id",
    "nrStatus",
    "run_num",
    "seq_num",
    "delta",
    "variation_relative",
    "slope_brut",
    "slope_lisse",
    "seuil",
]

# Save feature names
feature_names_path = os.path.join(model_dir, "feature_names.pkl")
joblib.dump(feature_names, feature_names_path)
print(f"Saved feature names to: {feature_names_path}")

# Create dummy scaler
scaler = StandardScaler()
# Fit with random data
random_data = np.random.randn(100, len(feature_names))
scaler.fit(random_data)
scaler_path = os.path.join(model_dir, "scaler.gz")
joblib.dump(scaler, scaler_path)
print(f"Saved scaler to: {scaler_path}")

# Create dummy encoder
encoder = OneHotEncoder(sparse_output=False)
categorical_data = np.array(
    [["Good", "Walking"], ["Medium", "Driving"], ["Poor", "Indoor"]]
)
encoder.fit(categorical_data)
encoder_path = os.path.join(model_dir, "encoder.gz")
joblib.dump(encoder, encoder_path)
print(f"Saved encoder to: {encoder_path}")

# Create dummy model
if KERAS_AVAILABLE:
    # Create a simple Keras model
    inputs = keras.Input(shape=(len(feature_names),))
    x = keras.layers.Dense(10, activation="relu")(inputs)
    outputs = keras.layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")

    # Save model
    model_path = os.path.join(model_dir, "throughput_model.keras")
    model.save(model_path)
    print(f"Saved Keras model to: {model_path}")
else:
    # Create a simple joblib model
    from sklearn.linear_model import LinearRegression

    model = LinearRegression()
    model.fit(random_data, np.random.randn(100, 1))
    model_path = os.path.join(model_dir, "throughput_model.joblib")
    joblib.dump(model, model_path)
    print(f"Saved joblib model to: {model_path}")

print("Done! Dummy ML models created successfully.")
