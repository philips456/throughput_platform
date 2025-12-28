"""
Script: evaluate_model.py
Objectif: Évaluer le modèle sauvegardé sur un jeu de test
"""

import pandas as pd
import joblib
import pickle
from tensorflow import keras
from sklearn.metrics import mean_squared_error, mean_absolute_error
from ingest_data import ingest_data


def evaluate():
    df = ingest_data()
    y = df["throughput"]
    X = df.drop(columns=["throughput"])

    scaler = joblib.load("ml_models/scaler.gz")
    X_scaled = scaler.transform(X)

    model = keras.models.load_model("ml_models/throughput_model.keras")

    y_pred = model.predict(X_scaled)
    rmse = mean_squared_error(y, y_pred, squared=False)
    mae = mean_absolute_error(y, y_pred)

    print(f"[INFO] RMSE: {rmse:.4f}")
    print(f"[INFO] MAE : {mae:.4f}")


if __name__ == "__main__":
    evaluate()
