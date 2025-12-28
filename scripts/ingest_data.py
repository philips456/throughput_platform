"""
Script: ingest_data.py
Objectif: Charger et préparer les données depuis mm-5G-enriched.csv
"""

import pandas as pd


def ingest_data(
    path=r"D:\SEMESTER 2\PI\5G_throughput_prediction\data\mm-5G-enriched.csv",
):
    df = pd.read_csv(path)
    print(f"[INFO] {len(df)} lignes chargées depuis {path}")
    return df


if __name__ == "__main__":
    ingest_data()
