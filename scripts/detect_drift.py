"""
Script: detect_drift.py
Objectif: Détecter un drift de données sur les features
"""

import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from ingest_data import ingest_data


def detect_drift():
    df = ingest_data()
    new_features = list(df.drop(columns=["throughput"]).columns)

    with open("ml_models/feature_names.pkl", "rb") as f:
        original_features = pickle.load(f)

    drift = set(original_features).symmetric_difference(set(new_features))

    if drift:
        print(f"[ALERTE] Drift détecté: {drift}")
    else:
        print("[INFO] Aucun drift détecté.")


if __name__ == "__main__":
    detect_drift()
