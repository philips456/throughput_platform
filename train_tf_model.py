#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import mlflow
import mlflow.tensorflow
from preprocess import load_data, build_sequences
from sklearn.model_selection import train_test_split
from keras import layers, models, callbacks
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import psutil  # For system metrics

# ──────────────────────────────────────────────────────────────────
# CONFIGURATION
DATA_PATH = "/mnt/c/Users/phili/Downloads/PI/PI 2/data/mm-5G-enriched.csv"
SEQ_LEN = 10
MODEL_DIR = "/mnt/c/Users/phili/Downloads/PI/PI 2/ml_models"
OUTPUTS_DIR = "/mnt/c/Users/phili/Downloads/PI/PI 2/outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────────
# PREPROCESSING FUNCTIONS
# ──────────────────────────────────────────────────────────────────
def preprocess_data(df):
    """Handle categorical encoding and feature engineering"""
    # Encode categorical features
    categorical_cols = ["abstractSignalStr", "mobility_mode"]
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    encoded_features = encoder.fit_transform(df[categorical_cols])

    # Create DataFrame with encoded features
    encoded_df = pd.DataFrame(
        encoded_features, columns=encoder.get_feature_names_out(categorical_cols)
    )

    # Merge with original data
    processed_df = pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

    # Save encoder
    joblib.dump(encoder, os.path.join(MODEL_DIR, "encoder.gz"))
    return processed_df, encoder


def scale_features(X):
    """Scale features using StandardScaler"""
    scaler = StandardScaler()
    original_shape = X.shape
    X = scaler.fit_transform(X.reshape(-1, X.shape[-1]))
    return X.reshape(original_shape), scaler


# ──────────────────────────────────────────────────────────────────
# SYSTEM METRICS FUNCTION
# ──────────────────────────────────────────────────────────────────
def log_system_metrics():
    """Log system metrics such as CPU and Memory usage"""
    cpu_usage = psutil.cpu_percent(interval=1)  # CPU usage in percentage
    memory = psutil.virtual_memory()  # Memory info

    # Log metrics
    mlflow.log_metric("cpu_usage", cpu_usage)
    mlflow.log_metric("memory_used", memory.used)
    mlflow.log_metric("memory_free", memory.available)
    mlflow.log_metric("memory_total", memory.total)

    # Print system metrics
    print(f"System Metrics - CPU Usage: {cpu_usage}%")
    print(f"System Metrics - Memory Used: {memory.used / (1024 * 1024):.2f} MB")
    print(f"System Metrics - Memory Free: {memory.available / (1024 * 1024):.2f} MB")
    print(f"System Metrics - Total Memory: {memory.total / (1024 * 1024):.2f} MB")


# ──────────────────────────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────
with mlflow.start_run():
    # ==================================================================
    # DATA PREPARATION
    # ==================================================================
    print("🔹 Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    df_processed, encoder = preprocess_data(df)

    print("🔹 Building sequences...")
    X, y_raw, y_smooth, runs, feature_names = build_sequences(df_processed, SEQ_LEN)

    # Save feature names
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

    print("🔹 Scaling features...")
    X, scaler = scale_features(X)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.gz"))

    # ==================================================================
    # DATA SPLITTING
    # ==================================================================
    print("🔹 Splitting data...")
    (
        X_train,
        X_val,
        y_raw_train,
        y_raw_val,
        y_smooth_train,
        y_smooth_val,
        runs_train,
        runs_val,
    ) = train_test_split(
        X, y_raw, y_smooth, runs, test_size=0.2, random_state=42, shuffle=False
    )

    # ==================================================================
    # MODEL ARCHITECTURE
    # ==================================================================
    print("🔹 Building model...")
    n_features = X.shape[2]

    inputs = layers.Input(shape=(SEQ_LEN, n_features))
    x = layers.Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64)(x)

    raw_output = layers.Dense(1, name="raw")(x)
    smooth_output = layers.Dense(1, name="smooth")(x)

    model = models.Model(inputs=inputs, outputs=[raw_output, smooth_output])
    model.compile(optimizer="adam", loss="mse")

    # ==================================================================
    # TRAINING
    # ==================================================================
    print("🔹 Training model...")
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train,
        [y_raw_train, y_smooth_train],
        validation_data=(X_val, [y_raw_val, y_smooth_val]),
        epochs=50,
        batch_size=32,
        callbacks=[early_stopping],
    )

    # ==================================================================
    # SAVING ARTIFACTS AND LOGGING METRICS
    # ==================================================================
    print("🔹 Saving final model...")
    model.save(os.path.join(MODEL_DIR, "throughput_model.keras"))

    # Log system metrics
    log_system_metrics()

    # Log metrics
    pred_raw, pred_smooth = model.predict(X_val)
    pred_raw = pred_raw.squeeze()
    pred_smooth = pred_smooth.squeeze()
    pred_final = np.where(
        np.abs(pred_raw - y_raw_val) < np.abs(pred_smooth - y_raw_val),
        pred_raw,
        pred_smooth,
    )

    # RMSE calculations
    rmse_raw = np.sqrt(mean_squared_error(y_raw_val, pred_raw))
    rmse_smooth = np.sqrt(mean_squared_error(y_smooth_val, pred_smooth))
    rmse_final = np.sqrt(mean_squared_error(y_raw_val, pred_final))

    # MAE calculation
    mae_raw = mean_absolute_error(y_raw_val, pred_raw)
    mae_smooth = mean_absolute_error(y_smooth_val, pred_smooth)
    mae_final = mean_absolute_error(y_raw_val, pred_final)

    # R² calculation
    r2_raw = r2_score(y_raw_val, pred_raw)
    r2_smooth = r2_score(y_smooth_val, pred_smooth)
    r2_final = r2_score(y_raw_val, pred_final)

    # Log metrics to MLflow
    mlflow.log_metric("rmse_raw", rmse_raw)
    mlflow.log_metric("rmse_smooth", rmse_smooth)
    mlflow.log_metric("rmse_final", rmse_final)
    mlflow.log_metric("mae_raw", mae_raw)
    mlflow.log_metric("mae_smooth", mae_smooth)
    mlflow.log_metric("mae_final", mae_final)
    mlflow.log_metric("r2_raw", r2_raw)
    mlflow.log_metric("r2_smooth", r2_smooth)
    mlflow.log_metric("r2_final", r2_final)

    # Print metrics to console
    print("Metrics for Model Evaluation:")
    print(f"RMSE (Raw): {rmse_raw:.4f}")
    print(f"RMSE (Smooth): {rmse_smooth:.4f}")
    print(f"RMSE (Final): {rmse_final:.4f}")
    print(f"MAE (Raw): {mae_raw:.4f}")
    print(f"MAE (Smooth): {mae_smooth:.4f}")
    print(f"MAE (Final): {mae_final:.4f}")
    print(f"R² (Raw): {r2_raw:.4f}")
    print(f"R² (Smooth): {r2_smooth:.4f}")
    print(f"R² (Final): {r2_final:.4f}")

    # Add tags to the model
    mlflow.set_tag("model_version", "v1.0")
    mlflow.set_tag("dataset_used", "mm-5G-enriched")
    mlflow.set_tag("model_type", "CNN + LSTM")

    # Register model
    run = mlflow.active_run()  # Get the current active run
    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model", name="SeqModel_Throughput"
    )

    print("✅ Training complete! Artifacts saved to:")
    print(f"📂 {MODEL_DIR}")
    print("- encoder.gz")
    print("- scaler.gz")
    print("- feature_names.pkl")
    print("- throughput_model.keras")
