import os
import traceback

try:
    from tensorflow.keras.models import load_model
except ImportError:
    print("❌ TensorFlow is not installed.")
    exit(1)

# 📍 Chemin absolu vers le modèle
# 📍 Chemin absolu vers le modèle
model_path = "/mnt/d/SEMESTER 2/PI/5G_throughput_prediction/ml_models/throughput_model.keras"


def test_model_loading():
    print(f"🔍 Trying to load model from: {model_path}")

    if not os.path.exists(model_path):
        print("❌ Model file does not exist.")
        return

    try:
        model = load_model(model_path)
        print("✅ Model loaded successfully.")
        model.summary()  # Affiche l’architecture du modèle
    except Exception as e:
        print("❌ Failed to load the model.")
        print("Error:", e)
        traceback.print_exc()


if __name__ == "__main__":
    test_model_loading()
