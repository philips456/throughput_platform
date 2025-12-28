"""
Script: deploy_model.py
Objectif: Déployer le modèle dans un répertoire de production
"""

import shutil
import os


def deploy():
    files = ["throughput_model.keras", "scaler.gz", "feature_names.pkl"]
    src = "ml_models"
    dest = "prod_models"
    os.makedirs(dest, exist_ok=True)

    for file in files:
        shutil.copy(os.path.join(src, file), os.path.join(dest, file))
        print(f"[INFO] {file} déployé vers {dest}/")


if __name__ == "__main__":
    deploy()
