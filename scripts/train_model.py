import os
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.keras
import tensorflow as tf
import matplotlib.pyplot as plt

from keras import layers, models, callbacks
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Load dataset
path = r"D:\SEMESTER 2\PI\5G_throughput_prediction\data\mm-5G-enriched.csv"
df = pd.read_csv(path)
df.drop(columns=["debit_class"], inplace=True, errors="ignore")

# Define features and target
SEQ_LEN = 10
targets = ["debit_brut", "debit_lisse"]
features = df.select_dtypes(include=[np.number]).columns.drop(targets).tolist()

# Build sequences
X, y_raw, y_smooth = [], [], []
for run_id, grp in df.groupby("run_num"):
    grp = grp.reset_index(drop=True)
    for i in range(len(grp) - SEQ_LEN):
        X.append(grp.loc[i : i + SEQ_LEN - 1, features].values)
        y_raw.append(grp.loc[i + SEQ_LEN, "debit_brut"])
        y_smooth.append(grp.loc[i + SEQ_LEN, "debit_lisse"])

X = np.stack(X)
y_raw = np.array(y_raw)
y_smooth = np.array(y_smooth)

# Scale input
ns, _, nf = X.shape
scaler = MinMaxScaler()
X_flat = scaler.fit_transform(X.reshape(-1, nf))
X = X_flat.reshape(ns, SEQ_LEN, nf)
joblib.dump(scaler, "models/scaler.gz")

# Split
X_train, X_val, y_raw_train, y_raw_val, y_smooth_train, y_smooth_val = train_test_split(
    X, y_raw, y_smooth, test_size=0.2, random_state=42, shuffle=False
)

# Model
input_layer = layers.Input(shape=(SEQ_LEN, nf))
x = layers.Conv1D(32, 3, padding="same", activation="relu")(input_layer)
x = layers.Conv1D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling1D(2)(x)
x = layers.LSTM(64)(x)
raw_output = layers.Dense(1, name="raw")(x)
smooth_output = layers.Dense(1, name="smooth")(x)
model = models.Model(inputs=input_layer, outputs=[raw_output, smooth_output])
model.compile(optimizer="adam", loss="mse")

# MLflow tracking
mlflow.set_experiment("5G_Throughput_SeqModel")

with mlflow.start_run() as run:
    es = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True
    )
    history = model.fit(
        X_train,
        [y_raw_train, y_smooth_train],
        validation_data=(X_val, [y_raw_val, y_smooth_val]),
        epochs=100,
        batch_size=32,
        callbacks=[es],
    )

    model.save("models/throughput_model.keras")
    mlflow.keras.log_model(model, "model")

    # Metrics
    pred_raw, pred_smooth = model.predict(X_val)
    pred_raw = pred_raw.squeeze()
    pred_smooth = pred_smooth.squeeze()
    pred_final = np.where(
        np.abs(pred_raw - y_raw_val) < np.abs(pred_smooth - y_raw_val),
        pred_raw,
        pred_smooth,
    )

    rmse_raw = np.sqrt(mean_squared_error(y_raw_val, pred_raw))
    rmse_smooth = np.sqrt(mean_squared_error(y_smooth_val, pred_smooth))
    rmse_final = np.sqrt(mean_squared_error(y_raw_val, pred_final))
    mae_final = mean_absolute_error(y_raw_val, pred_final)
    r2_final = r2_score(y_raw_val, pred_final)

    mlflow.log_metric("rmse_raw", rmse_raw)
    mlflow.log_metric("rmse_smooth", rmse_smooth)
    mlflow.log_metric("rmse_final", rmse_final)
    mlflow.log_metric("mae_final", mae_final)
    mlflow.log_metric("r2_final", r2_final)

    # Add tags to the model
    mlflow.set_tag("model_version", "v1.0")
    mlflow.set_tag("dataset_used", "mm-5G-enriched")
    mlflow.set_tag("model_type", "CNN + LSTM")

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.title("Training vs. Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    loss_plot_path = "outputs/loss_plot.png"
    plt.savefig(loss_plot_path)
    mlflow.log_artifact(loss_plot_path)

    # Register model
    mlflow.register_model(
        model_uri=f"runs:/{run.info.run_id}/model", name="SeqModel_Throughput"
    )
