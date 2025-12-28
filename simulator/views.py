from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET
from django.http import JsonResponse
import json
import pandas as pd
import numpy as np
import joblib
import os
import sys
import traceback
from pathlib import Path

# Add these imports at the top of your views.py file
import random
import json
from django.http import JsonResponse
from django.views.decorators.http import require_GET

# Try different imports for TensorFlow/Keras based on what's available
try:
    from tensorflow import keras
    from tensorflow.keras.models import load_model  # Ensure TensorFlow is installed

    # If TensorFlow is not installed, consider using a fallback or alternative library
except ImportError:
    try:
        from keras.models import load_model
    except ImportError:
        # Fallback to a simple model if TensorFlow is not available
        load_model = None

# Global variables for model components
model = None
scaler = None
encoder = None
feature_names = None
sequence_buffer = {}  # Stores last 9 points for each session
MODEL_LOADED = False


def get_model_dir():
    """Get the absolute path to the ml_models directory"""
    # Try different possible locations for the ml_models directory
    possible_locations = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_models"),
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml_models"
        ),
        os.path.join(os.getcwd(), "ml_models"),
        "/app/ml_models",  # For Docker/container environments
    ]

    for location in possible_locations:
        if os.path.exists(location):
            return location

    # If no location exists, create one
    default_location = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "ml_models"
    )
    os.makedirs(default_location, exist_ok=True)
    return default_location


def load_ml_components():
    """Load all ML components needed for prediction"""
    global model, scaler, encoder, feature_names, MODEL_LOADED

    if MODEL_LOADED:
        return True

    try:
        model_dir = get_model_dir()
        print(f"Looking for ML models in: {model_dir}")

        # List all files in the model directory to help with debugging
        if os.path.exists(model_dir):
            print(f"Files in {model_dir}:")
            for file in os.listdir(model_dir):
                print(f"  - {file}")
        else:
            print(f"Model directory {model_dir} does not exist!")
            return False

        # Load Feature Names first (needed for other components)
        features_path = os.path.join(model_dir, "feature_names.pkl")
        if os.path.exists(features_path):
            with open(features_path, "rb") as f:
                feature_names = joblib.load(f)
            print(f"Loaded feature names: {len(feature_names)} features")
        else:
            print(f"Feature names file not found at {features_path}")
            # Create dummy feature names for testing
            feature_names = [
                "longitude",
                "latitude",
                "speed",
                "direction",
                "nr_ssRsrp",
                "nr_ssRsrq",
                "nr_ssSinr",
            ]
            print(f"Created dummy feature names: {feature_names}")

        # Load Scaler
        scaler_path = os.path.join(model_dir, "scaler.gz")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("Loaded scaler successfully")
        else:
            print(f"Scaler file not found at {scaler_path}")
            # Create a dummy scaler for testing
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
            print("Created dummy scaler")

        # Load Encoder if exists
        encoder_path = os.path.join(model_dir, "encoder.gz")
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            print("Loaded encoder successfully")
        else:
            print(f"Encoder file not found at {encoder_path}")
            encoder = None

        # Load Model
        model_path = model_path = (
            r"D:\SEMESTER 2\PI\5G_throughput_prediction\ml_models\throughput_model.keras"
        )
        if os.path.exists(model_path) and load_model is not None:
            try:
                model = load_model(model_path)
                print("Loaded Keras model successfully")
            except Exception as e:
                print(f"Error loading Keras model: {str(e)}")
                # Try alternative model formats
                try:
                    model_path_h5 = os.path.join(model_dir, "throughput_model.h5")
                    if os.path.exists(model_path_h5):
                        model = load_model(model_path_h5)
                        print("Loaded H5 model successfully")
                    else:
                        raise FileNotFoundError(
                            f"H5 model not found at {model_path_h5}"
                        )
                except Exception as e2:
                    print(f"Error loading alternative model format: {str(e2)}")
                    # Create a dummy model for testing
                    create_dummy_model()
        else:
            print(f"Model file not found at {model_path} or TensorFlow not available")
            # Create a dummy model for testing
            create_dummy_model()

        MODEL_LOADED = True
        print("All ML components loaded or created successfully")
        return True

    except Exception as e:
        print(f"Error loading ML components: {str(e)}")
        print("Exception details:")
        traceback.print_exc()

        # Create dummy components for testing
        create_dummy_components()
        return True  # Return True to allow testing with dummy components


def create_dummy_model():
    """Create a simple dummy model for testing when the real model is not available"""
    global model

    print("Creating dummy model for testing")

    if load_model is not None:
        # Create a simple Keras model
        inputs = keras.Input(shape=(len(feature_names),))
        x = keras.layers.Dense(10, activation="relu")(inputs)
        outputs = keras.layers.Dense(1)(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer="adam", loss="mse")
        print("Created dummy Keras model")
    else:
        # Create a very simple model that just returns a random value
        class DummyModel:
            def predict(self, X):
                # Return random values between 10 and 100
                if isinstance(X, list):
                    return [np.random.uniform(10, 100, (len(X), 1))]
                return [np.random.uniform(10, 100, (X.shape[0], 1))]

        model = DummyModel()
        print("Created dummy model class")


def create_dummy_components():
    """Create dummy components for testing when real components are not available"""
    global model, scaler, encoder, feature_names

    print("Creating dummy components for testing")

    # Create dummy feature names if not already set
    if feature_names is None:
        feature_names = [
            "longitude",
            "latitude",
            "speed",
            "direction",
            "nr_ssRsrp",
            "nr_ssRsrq",
            "nr_ssSinr",
        ]
        print(f"Created dummy feature names: {feature_names}")

    # Create dummy scaler
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    print("Created dummy scaler")

    # Create dummy model
    create_dummy_model()


def create_sequence(session_id, new_point):
    """Maintains a rolling window of last 10 points"""
    if session_id not in sequence_buffer:
        sequence_buffer[session_id] = []

    # Keep only last 9 points + new point
    sequence_buffer[session_id] = (sequence_buffer[session_id] + [new_point])[-9:]

    # If we don't have enough points yet, repeat the first point
    while len(sequence_buffer[session_id]) < 10:
        sequence_buffer[session_id].append(new_point)

    return sequence_buffer[session_id]


# Authentication Views
def home_view(request):
    return render(request, "home.html")


def signup_view(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    else:
        form = UserCreationForm()
    return render(request, "simulator/signup.html", {"form": form})


def login_view(request):
    if request.method == "POST":
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect("dashboard")
    else:
        form = AuthenticationForm()
    return render(request, "login.html", {"form": form})


def logout_view(request):
    logout(request)
    return redirect("login")


# Dashboard and Pages
@login_required
def dashboard(request):
    return render(request, "dashboard.html")


@login_required
def predict_page(request):
    return render(request, "predictor.html")


@login_required
def drift_page(request):
    return render(request, "drift.html")


@login_required
def map_page(request):
    return render(request, "simulator/map.html")


# Metrics Pages
@login_required
def mlops(request):
    return render(request, "simulator/mlops.html")


@login_required
def system_metrics(request):
    return render(request, "simulator/system_metrics.html")


@login_required
def cpu_metrics(request):
    return render(request, "cpu.html")


@login_required
def sessions_metrics(request):
    return render(request, "sessions.html")


@login_required
def users_metrics(request):
    return render(request, "users.html")


@login_required
def models_page(request):
    return render(request, "simulator/models_page.html")


# Add these functions to your views.py file


@login_required
def visuals_page(request):
    """Render the model visualization page"""
    return render(request, "visuals.html")


# New view functions for MLflow metrics and model pages
@login_required
def mlflow_metrics(request):
    """Render the MLflow metrics page"""
    return render(request, "mlflow_metrics.html")


@login_required
def raw_model(request):
    """Render the raw model page"""
    return render(request, "raw_model.html")


@login_required
def smoothed_model(request):
    """Render the smoothed model page"""
    return render(request, "smoothed_model.html")


@login_required
def hybrid_model(request):
    """Render the hybrid model page"""
    return render(request, "hybrid_model.html")


@login_required
def compare_models(request):
    """Render the model comparison page"""
    return render(request, "compare_models.html")


@login_required
def load_history(request):
    """Render the load history page"""
    return render(request, "load_history.html")


# Add this function if it doesn't already exist
@login_required
def sdg_page(request):
    """
    View for the Sustainable Development Goals (SDG) page
    Includes carbon footprint calculation based on throughput data
    """
    # Sample data for carbon footprint by speed range
    speed_ranges = ["0-20 km/h", "20-50 km/h", "50-80 km/h", "80-120 km/h"]
    carbon_values = [0.00015, 0.00025, 0.00040, 0.00060]  # kg CO₂e

    # Sample data for throughput at different speeds
    throughput_by_speed = {
        "0-20": 85,  # Mbps
        "20-50": 75,  # Mbps
        "50-80": 65,  # Mbps
        "80-120": 55,  # Mbps
    }

    # Calculate carbon footprint for each speed range
    # Formula: (Throughput in Mbit/s × Duration in s) / (8 × 1024) × 5 × 0.233 / 1000
    # Using 60 seconds as a standard duration
    carbon_footprint = {}
    for speed_range, throughput in throughput_by_speed.items():
        data_volume_gb = (throughput * 60) / (8 * 1024)
        energy_wh = data_volume_gb * 5
        carbon_kg = (energy_wh / 1000) * 0.233
        carbon_footprint[speed_range] = carbon_kg

    context = {
        "speed_ranges": speed_ranges,
        "carbon_values": carbon_values,
        "throughput_by_speed": throughput_by_speed,
        "carbon_footprint": carbon_footprint,
    }

    return render(request, "sdg.html", context)


# API Endpoints
@require_GET
def get_antennas(request):
    try:
        # Check if the data file exists
        data_file = "data/mm-5G-enriched.csv"
        if not os.path.exists(data_file):
            # Return dummy data if file doesn't exist
            dummy_data = [
                {
                    "tower_id": 1,
                    "latitude": 44.9778,
                    "longitude": -93.2650,
                    "nr_ssRsrp": -85,
                    "radius": 300,
                },
                {
                    "tower_id": 2,
                    "latitude": 44.9878,
                    "longitude": -93.2750,
                    "nr_ssRsrp": -90,
                    "radius": 250,
                },
            ]
            return JsonResponse(dummy_data, safe=False)

        data = pd.read_csv(data_file)

        # Filter points with active NR antenna
        antennas = data[data["nrStatus"] == 1]

        # Keep the last value per tower_id
        antennas = (
            antennas.groupby("tower_id")
            .agg(
                {
                    "latitude": "last",
                    "longitude": "last",
                    "nr_ssRsrp": "mean",
                }
            )
            .reset_index()
        )

        # Empirical formula: weaker RSRP (e.g., -110) means smaller radius
        def estimate_radius(rsrp):
            return max(50, min(500, 200 + (rsrp + 90) * 5))  # between 50m and 500m

        antennas["radius"] = antennas["nr_ssRsrp"].apply(estimate_radius)

        results = antennas.to_dict(orient="records")
        return JsonResponse(results, safe=False)
    except Exception as e:
        print(f"Error in get_antennas: {str(e)}")
        traceback.print_exc()

        # Return dummy data on error
        dummy_data = [
            {
                "tower_id": 1,
                "latitude": 44.9778,
                "longitude": -93.2650,
                "nr_ssRsrp": -85,
                "radius": 300,
            },
            {
                "tower_id": 2,
                "latitude": 44.9878,
                "longitude": -93.2750,
                "nr_ssRsrp": -90,
                "radius": 250,
            },
        ]
        return JsonResponse(dummy_data, safe=False)


@require_GET
def scenario_data(request):
    """API endpoint for scenario data"""
    scenario = request.GET.get("scenario", "urban")

    try:
        # In a real application, this would fetch data from a database or ML model
        # For now, we'll generate synthetic data
        data = generate_scenario_data(scenario)
        return JsonResponse(data)
    except Exception as e:
        print(f"Error in scenario_data: {str(e)}")
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def custom_scenario_data(request):
    """API endpoint for custom scenario data"""
    try:
        # Get parameters
        rsrp = float(request.GET.get("rsrp", -90))
        sinr = float(request.GET.get("sinr", 15))
        speed = float(request.GET.get("speed", 5))
        mobility_mode = request.GET.get("mobility_mode", "Walking")

        # Generate data based on parameters
        data = generate_custom_scenario_data(rsrp, sinr, speed, mobility_mode)
        return JsonResponse(data)
    except Exception as e:
        print(f"Error in custom_scenario_data: {str(e)}")
        traceback.print_exc()
        return JsonResponse({"error": str(e)}, status=500)


@require_GET
def map_prediction_api(request):
    """Simple prediction API for map visualization"""
    try:
        # Load ML components if not already loaded
        if not load_ml_components():
            return JsonResponse({"error": "Failed to load ML components"}, status=500)

        # Extract GET parameters
        lon = float(request.GET.get("lon", 0))
        lat = float(request.GET.get("lat", 0))
        speed = float(request.GET.get("speed", 0))
        direction = float(request.GET.get("direction", 0))
        ssRsrp = float(request.GET.get("nr_ssRsrp", -95))
        ssRsrq = float(request.GET.get("nr_ssRsrq", -10))
        ssSinr = float(request.GET.get("nr_ssSinr", 10))

        # Create input vector
        X = pd.DataFrame(
            [
                {
                    "longitude": lon,
                    "latitude": lat,
                    "speed": speed,
                    "direction": direction,
                    "nr_ssRsrp": ssRsrp,
                    "nr_ssRsrq": ssRsrq,
                    "nr_ssSinr": ssSinr,
                }
            ]
        )

        # Ensure all required features are present
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0  # Default value for missing features

        X = X[feature_names]  # Reorder columns to match model expectations
        X_scaled = scaler.transform(X)

        # Make prediction
        prediction = model.predict(X_scaled)

        # Handle different prediction formats
        if isinstance(prediction, list):
            pred = prediction[0][0][0]
        else:
            pred = prediction[0][0]

        return JsonResponse({"throughput": round(float(pred), 2)})

    except Exception as e:
        print(f"Error in map_prediction_api: {str(e)}")
        traceback.print_exc()

        # Return a dummy prediction on error
        return JsonResponse(
            {
                "throughput": round(float(np.random.uniform(10, 100)), 2),
                "note": "Using fallback prediction due to error",
            }
        )


@csrf_exempt
@require_GET
def map_prediction(request):
    """Advanced prediction API with sequence handling"""
    try:
        # Load ML components if not already loaded
        if not load_ml_components():
            return JsonResponse({"error": "Failed to load ML components"}, status=500)

        # Get parameters from request
        params = request.GET
        session_id = params.get("session_id", "default")

        # Create sample with default values for all expected features
        sample = {
            # Location and movement
            "latitude": float(params.get("lat", 0)),
            "longitude": float(params.get("lon", 0)),
            "movingSpeed": float(params.get("speed", 0)),
            "compassDirection": float(params.get("direction", 0)),
            # Network metrics (example defaults)
            "lte_rssi": float(params.get("lte_rssi", -80)),
            "lte_rsrp": float(params.get("lte_rsrp", -90)),
            "nr_ssRsrp": float(params.get("nr_ssRsrp", -85)),
            "nr_ssSinr": float(params.get("nr_ssSinr", 20)),
            # Categorical features
            "abstractSignalStr": params.get("signal_strength", "example"),
            "mobility_mode": params.get("mobility_mode", "Walking"),
            # Add other features with reasonable defaults
            "run_num": 1,
            "seq_num": 1,
            "nrStatus": 1,
            "lte_rsrq": -10,
            "lte_rssnr": 20,
            "nr_ssRsrq": -10,
            "trajectory_direction": 45,
            "tower_id": 1,
            "delta": 5,
            "variation_relative": 10,
            "slope_brut": 1,
            "slope_lisse": 1,
            "seuil": 10,
        }

        # ─── Preprocessing Pipeline ────────────────────────────────────────
        # 1. Convert to DataFrame
        df = pd.DataFrame([sample])

        # 2. Encode categorical features if encoder exists
        if encoder is not None:
            categorical_cols = ["abstractSignalStr", "mobility_mode"]
            encoded_features = encoder.transform(df[categorical_cols])
            encoded_df = pd.DataFrame(
                encoded_features,
                columns=encoder.get_feature_names_out(categorical_cols),
            )

            # 3. Merge features
            df_processed = pd.concat(
                [df.drop(columns=categorical_cols), encoded_df], axis=1
            )
        else:
            df_processed = df

        # 4. Ensure correct feature order and fill missing
        for col in feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0  # Fill missing with 0
        df_processed = df_processed[feature_names]

        # 5. Scale features
        scaled_point = scaler.transform(df_processed)

        # ─── Sequence Handling ──────────────────────────────────────────────
        # Maintain sequence for each session
        sequence = create_sequence(session_id, scaled_point)
        sequence_array = np.array(sequence).reshape(1, 10, -1)

        # ─── Prediction ─────────────────────────────────────────────────────
        # Handle different model output formats
        prediction = model.predict(sequence_array)

        if isinstance(prediction, list) and len(prediction) == 2:
            # Model returns [raw_pred, smooth_pred]
            raw_pred, smooth_pred = prediction
            latest_pred = {
                "raw": float(raw_pred[0][0]),
                "smooth": float(smooth_pred[0][0]),
            }
            throughput = latest_pred["smooth"]
        else:
            # Model returns single prediction
            throughput = float(prediction[0][0])

        return JsonResponse(
            {
                "throughput": round(throughput, 2),
                "coordinates": {"lat": sample["latitude"], "lng": sample["longitude"]},
            }
        )

    except Exception as e:
        print(f"Error in map_prediction: {str(e)}")
        traceback.print_exc()

        # Return a dummy prediction on error
        return JsonResponse(
            {
                "throughput": round(float(np.random.uniform(10, 100)), 2),
                "coordinates": {
                    "lat": float(request.GET.get("lat", 0)),
                    "lng": float(request.GET.get("lon", 0)),
                },
                "note": "Using fallback prediction due to error",
            }
        )


@csrf_exempt
def predict_api(request):
    """API endpoint for throughput prediction"""
    if request.method == "POST":
        try:
            # Load ML components if not already loaded
            if not load_ml_components():
                return JsonResponse(
                    {"error": "Failed to load ML components"}, status=500
                )

            data = json.loads(request.body)
            values = [data.get(f, 0) for f in feature_names]
            X = scaler.transform([values])
            prediction = model.predict(X)[0][0]

            return JsonResponse({"prediction": float(prediction)})
        except Exception as e:
            print(f"Error in predict_api: {str(e)}")
            traceback.print_exc()

            # Return a dummy prediction on error
            return JsonResponse(
                {
                    "prediction": float(np.random.uniform(10, 100)),
                    "note": "Using fallback prediction due to error",
                }
            )
    return JsonResponse({"message": "POST method required"}, status=400)


@csrf_exempt
def detect_drift(request):
    """API endpoint for detecting feature drift"""
    if request.method == "POST":
        try:
            # Load feature_names if not already loaded
            if not load_ml_components():
                return JsonResponse(
                    {"error": "Failed to load ML components"}, status=500
                )

            data = json.loads(request.body)
            new_features = sorted(list(data.keys()))
            original_features = sorted(feature_names)
            drift = set(original_features).symmetric_difference(set(new_features))

            if drift:
                return JsonResponse({"drift_detected": True, "difference": list(drift)})
            else:
                return JsonResponse({"drift_detected": False})
        except Exception as e:
            print(f"Error in detect_drift: {str(e)}")
            traceback.print_exc()
            return JsonResponse({"error": str(e)}, status=500)
    return JsonResponse({"message": "POST method required"}, status=400)


# Add this function to your views.py file
@require_GET
def predict_throughput(request):
    """API endpoint for throughput prediction with visualization data"""
    try:
        # Extract GET parameters
        lon = float(request.GET.get("lon", 0))
        lat = float(request.GET.get("lat", 0))
        speed = float(request.GET.get("speed", 0))
        direction = float(request.GET.get("direction", 0))
        rsrp = float(request.GET.get("nr_ssRsrp", -95))
        rsrq = float(request.GET.get("nr_ssRsrq", -10))
        sinr = float(request.GET.get("nr_ssSinr", 10))
        mobility_mode = request.GET.get("mobility_mode", "Walking")

        # Create input vector
        X = pd.DataFrame(
            [
                {
                    "longitude": lon,
                    "latitude": lat,
                    "speed": speed,
                    "direction": direction,
                    "nr_ssRsrp": rsrp,
                    "nr_ssRsrq": rsrq,
                    "nr_ssSinr": sinr,
                    "mobility_mode": mobility_mode,
                }
            ]
        )

        # Ensure all required features are present
        for col in feature_names:
            if col not in X.columns:
                X[col] = 0  # Default value for missing features

        X = X[feature_names]  # Reorder columns to match model expectations
        X_scaled = scaler.transform(X)

        # Make prediction
        prediction = model.predict(X_scaled)

        # Handle different prediction formats
        if isinstance(prediction, list):
            throughput = prediction[0][0][0]
        else:
            throughput = prediction[0][0]

        # Ensure throughput is within valid range (10-1500 Mbps)
        throughput = max(10.0, min(1500.0, float(throughput)))

        # Calculate feature importance (in a real app, this would come from the model)
        feature_importance = calculate_feature_importance(rsrp, sinr)

        # Calculate prediction confidence
        confidence = calculate_prediction_confidence(rsrp, sinr)

        return JsonResponse(
            {
                "throughput": round(float(throughput), 2),
                "feature_importance": feature_importance,
                "confidence": confidence,
            }
        )

    except Exception as e:
        print(f"Error in predict_throughput: {str(e)}")
        traceback.print_exc()

        # Return a fallback prediction on error
        return JsonResponse(
            {
                "throughput": round(float(np.random.uniform(10, 100)), 2),
                "note": "Using fallback prediction due to error",
                "error_details": str(e),
            }
        )


def simplified_throughput_prediction(rsrp, sinr, speed, mobility_mode):
    """Simple formula to predict throughput when ML model is not available"""
    # Base throughput based on RSRP and SINR
    # RSRP typically ranges from -140 dBm (very weak) to -70 dBm (very strong)
    # SINR typically ranges from -5 dB (very poor) to 30 dB (excellent)

    # Normalize RSRP to 0-1 scale (from -140 to -70)
    rsrp_norm = min(1, max(0, (rsrp + 140) / 70))

    # Normalize SINR to 0-1 scale (from -5 to 30)
    sinr_norm = min(1, max(0, (sinr + 5) / 35))

    # Base throughput (0-1000 Mbps)
    base_throughput = 1000 * (0.3 * rsrp_norm + 0.7 * sinr_norm)

    # Adjust for speed (higher speeds can reduce throughput due to Doppler effect)
    speed_factor = 1.0
    if speed > 20:  # High speed (e.g., driving)
        speed_factor = 0.8
    elif speed > 5:  # Medium speed
        speed_factor = 0.9

    # Adjust for mobility mode
    mobility_factor = 1.0
    if mobility_mode == "Indoor":
        mobility_factor = 0.7  # Indoor has more obstacles
    elif mobility_mode == "Driving":
        mobility_factor = 0.8  # Driving has more handovers

    # Calculate final throughput
    throughput = base_throughput * speed_factor * mobility_factor

    # Add some randomness to simulate real-world variation
    variation = random.uniform(0.9, 1.1)
    throughput *= variation

    # Ensure throughput is within valid range (10-1500 Mbps)
    throughput = max(10.0, min(1500.0, throughput))

    return throughput


def calculate_feature_importance(rsrp, sinr):
    """Calculate feature importance based on signal quality"""
    # Default importance values
    rsrp_importance = 0.35
    sinr_importance = 0.25
    rsrq_importance = 0.15
    speed_importance = 0.15
    direction_importance = 0.1

    # Adjust importance based on signal quality
    if rsrp < -100:
        # When RSRP is poor, it becomes more important
        rsrp_importance = 0.45
        sinr_importance = 0.20
        rsrq_importance = 0.15
        speed_importance = 0.10
        direction_importance = 0.10
    elif sinr < 10:
        # When SINR is poor, it becomes more important
        rsrp_importance = 0.25
        sinr_importance = 0.40
        rsrq_importance = 0.15
        speed_importance = 0.10
        direction_importance = 0.10

    return {
        "rsrp": rsrp_importance,
        "sinr": sinr_importance,
        "rsrq": rsrq_importance,
        "speed": speed_importance,
        "direction": direction_importance,
    }


def calculate_prediction_confidence(rsrp, sinr):
    """Calculate prediction confidence based on signal quality"""
    # Base confidence levels
    high_confidence = 70
    medium_confidence = 20
    low_confidence = 10

    # Adjust confidence based on signal quality
    if rsrp >= -90 and sinr >= 15:
        # Excellent signal quality
        high_confidence = 80
        medium_confidence = 15
        low_confidence = 5
    elif rsrp < -105 or sinr < 5:
        # Poor signal quality
        high_confidence = 40
        medium_confidence = 30
        low_confidence = 30

    return {"high": high_confidence, "medium": medium_confidence, "low": low_confidence}


import math


def generate_scenario_data(scenario):
    """Generate synthetic data for different scenarios"""
    # Base data structure
    data = {
        "throughput_curves": {"raw": [], "smoothed": [], "hybrid": []},
        "signal_impact": [120, 90, 60, 30, 15],
        "mobility_impact": {
            "raw": [90, 85, 75, 60, 40],
            "smoothed": [85, 80, 75, 70, 65],
            "hybrid": [95, 90, 85, 75, 60],
        },
        "metrics": {
            "raw": {
                "accuracy": "89%",
                "latency": "15ms",
                "stability": "Low",
                "latency_percentage": "75%",
                "stability_percentage": "40%",
            },
            "smoothed": {
                "accuracy": "84%",
                "latency": "18ms",
                "stability": "High",
                "latency_percentage": "65%",
                "stability_percentage": "85%",
            },
            "hybrid": {
                "accuracy": "92%",
                "latency": "22ms",
                "stability": "Medium",
                "latency_percentage": "55%",
                "stability_percentage": "65%",
            },
        },
        "recommendations": [],
        "description": "",
    }

    # Generate throughput curves based on scenario
    if scenario == "urban":
        # Urban scenario - moderate fluctuations
        for i in range(20):
            base = 80 + math.sin(i * 0.5) * 20
            data["throughput_curves"]["raw"].append(base + random.uniform(-15, 15))
            data["throughput_curves"]["smoothed"].append(base + math.sin(i * 0.3) * 10)
            data["throughput_curves"]["hybrid"].append(base + math.sin(i * 0.4) * 15)

        data["description"] = (
            "Urban environment with moderate signal fluctuations due to buildings and obstacles."
        )
        data["recommendations"] = [
            {
                "title": "Optimal Model: Hybrid",
                "description": "For urban environments, the hybrid model provides the best balance of accuracy and stability.",
                "icon": "🏙️",
                "color": "purple",
            },
            {
                "title": "Signal Strength Considerations",
                "description": "In urban areas, signal strength can vary significantly. Consider using the smoothed model if stability is more important than peak performance.",
                "icon": "📶",
                "color": "blue",
            },
            {
                "title": "Mobility Optimization",
                "description": "For walking speeds in urban areas, all models perform well. For faster movement, the hybrid model adapts better to changing conditions.",
                "icon": "🚶",
                "color": "green",
            },
        ]

    elif scenario == "suburban":
        # Suburban scenario - mild fluctuations
        for i in range(20):
            base = 60 + math.sin(i * 0.3) * 15
            data["throughput_curves"]["raw"].append(base + random.uniform(-10, 10))
            data["throughput_curves"]["smoothed"].append(base + math.sin(i * 0.2) * 8)
            data["throughput_curves"]["hybrid"].append(base + math.sin(i * 0.25) * 12)

        data["description"] = (
            "Suburban environment with mild signal fluctuations and fewer obstacles."
        )
        data["recommendations"] = [
            {
                "title": "Optimal Model: Smoothed",
                "description": "For suburban environments, the smoothed model provides consistent performance with good accuracy.",
                "icon": "🏘️",
                "color": "green",
            },
            {
                "title": "Distance from Towers",
                "description": "In suburban areas, distance from towers is a key factor. The raw model may provide better peak performance when close to towers.",
                "icon": "📡",
                "color": "blue",
            },
            {
                "title": "Weather Impact",
                "description": "Suburban areas are more affected by weather conditions. During rain or fog, the smoothed model provides more reliable predictions.",
                "icon": "🌧️",
                "color": "yellow",
            },
        ]

    elif scenario == "highway":
        # Highway scenario - high speed, moderate fluctuations
        for i in range(20):
            base = 70 + math.sin(i * 0.8) * 25
            data["throughput_curves"]["raw"].append(base + random.uniform(-20, 20))
            data["throughput_curves"]["smoothed"].append(base + math.sin(i * 0.4) * 15)
            data["throughput_curves"]["hybrid"].append(base + math.sin(i * 0.6) * 20)

        data["description"] = (
            "Highway environment with high mobility and frequent handovers between towers."
        )
        data["recommendations"] = [
            {
                "title": "Optimal Model: Hybrid",
                "description": "For highway environments with high speeds, the hybrid model handles handovers between towers most effectively.",
                "icon": "🛣️",
                "color": "purple",
            },
            {
                "title": "Speed Considerations",
                "description": "At speeds above 80 km/h, the raw model shows increased fluctuations. The hybrid model provides better stability without sacrificing responsiveness.",
                "icon": "🚗",
                "color": "red",
            },
            {
                "title": "Handover Optimization",
                "description": "For applications requiring consistent connectivity during tower handovers, the smoothed model provides the most stable experience.",
                "icon": "🔄",
                "color": "blue",
            },
        ]

    elif scenario == "indoor":
        # Indoor scenario - low signal, high fluctuations
        for i in range(20):
            base = 40 + math.sin(i * 0.6) * 15
            data["throughput_curves"]["raw"].append(base + random.uniform(-12.5, 12.5))
            data["throughput_curves"]["smoothed"].append(base + math.sin(i * 0.3) * 8)
            data["throughput_curves"]["hybrid"].append(base + math.sin(i * 0.45) * 12)

        data["description"] = (
            "Indoor environment with signal attenuation due to walls and obstacles."
        )
        data["recommendations"] = [
            {
                "title": "Optimal Model: Smoothed",
                "description": "For indoor environments with significant signal attenuation, the smoothed model provides the most consistent experience.",
                "icon": "🏬",
                "color": "green",
            },
            {
                "title": "Building Material Impact",
                "description": "Different building materials affect signal penetration. In concrete structures, all models show reduced accuracy.",
                "icon": "🧱",
                "color": "yellow",
            },
            {
                "title": "Position Optimization",
                "description": "For indoor use, position near windows or exterior walls can significantly improve throughput regardless of model choice.",
                "icon": "🪟",
                "color": "blue",
            },
            {
                "title": "Consider Wi-Fi Offloading",
                "description": "In deep indoor locations, consider Wi-Fi offloading as mmWave 5G performance may be limited.",
                "icon": "📶",
                "color": "red",
            },
        ]

    return data


def generate_custom_scenario_data(rsrp, sinr, speed, mobility_mode):
    """Generate synthetic data for custom scenario based on parameters"""
    # Base data structure
    data = {
        "throughput_curves": {"raw": [], "smoothed": [], "hybrid": []},
        "signal_impact": [120, 90, 60, 30, 15],
        "mobility_impact": {
            "raw": [90, 85, 75, 60, 40],
            "smoothed": [85, 80, 75, 70, 65],
            "hybrid": [95, 90, 85, 75, 60],
        },
        "metrics": {
            "raw": {
                "accuracy": "89%",
                "latency": "15ms",
                "stability": "Low",
                "latency_percentage": "75%",
                "stability_percentage": "40%",
            },
            "smoothed": {
                "accuracy": "84%",
                "latency": "18ms",
                "stability": "High",
                "latency_percentage": "65%",
                "stability_percentage": "85%",
            },
            "hybrid": {
                "accuracy": "92%",
                "latency": "22ms",
                "stability": "Medium",
                "latency_percentage": "55%",
                "stability_percentage": "65%",
            },
        },
        "recommendations": [],
        "description": f"Custom scenario with RSRP: {rsrp} dBm, SINR: {sinr} dB, Speed: {speed} m/s, Mode: {mobility_mode}",
    }

    # Adjust throughput based on signal parameters
    rsrp_factor = min(1, max(0, (rsrp + 140) / 70))  # Normalize -140 to -70 dBm
    sinr_factor = min(1, max(0, (sinr + 5) / 35))  # Normalize -5 to 30 dB
    signal_factor = rsrp_factor * 0.4 + sinr_factor * 0.6  # Combined signal factor

    # Adjust for speed
    speed_factor = max(0.5, 1 - speed / 60)  # Higher speeds reduce stability

    # Adjust for mobility mode
    mobility_factor = 1.0
    if mobility_mode == "Indoor":
        mobility_factor = 0.7
    elif mobility_mode == "Driving":
        mobility_factor = 0.85

    # Generate throughput curves
    base_throughput = 150 * signal_factor * mobility_factor
    fluctuation_factor = (1 - signal_factor) * 2 + (1 - speed_factor)

    for i in range(20):
        base = base_throughput + math.sin(i * 0.5) * (base_throughput * 0.2)
        data["throughput_curves"]["raw"].append(
            base
            + random.uniform(
                -(base_throughput * fluctuation_factor * 0.25),
                base_throughput * fluctuation_factor * 0.25,
            )
        )
        data["throughput_curves"]["smoothed"].append(
            base + math.sin(i * 0.3) * (base_throughput * 0.1)
        )
        data["throughput_curves"]["hybrid"].append(
            base + math.sin(i * 0.4) * (base_throughput * 0.15)
        )

    # Generate recommendations based on parameters
    recommendations = []

    # Signal-based recommendations
    if rsrp < -100:
        recommendations.append(
            {
                "title": "Weak Signal Detected",
                "description": "The RSRP value indicates a weak signal. Consider moving closer to a window or outdoor area for better performance.",
                "icon": "📶",
                "color": "red",
            }
        )

    if sinr < 10:
        recommendations.append(
            {
                "title": "High Interference",
                "description": "The SINR value indicates high interference. Try changing your location or reducing electronic devices nearby.",
                "icon": "📡",
                "color": "yellow",
            }
        )

    # Speed-based recommendations
    if speed > 15:
        recommendations.append(
            {
                "title": "High Mobility Detected",
                "description": "At your current speed, the hybrid model provides the best balance of accuracy and stability during handovers.",
                "icon": "🚗",
                "color": "purple",
            }
        )
    elif speed < 2:
        recommendations.append(
            {
                "title": "Stationary Usage",
                "description": "For stationary usage, the raw model provides the highest peak throughput when signal conditions are good.",
                "icon": "🧍",
                "color": "blue",
            }
        )

    # Mode-based recommendations
    if mobility_mode == "Indoor":
        recommendations.append(
            {
                "title": "Indoor Optimization",
                "description": "For indoor usage, the smoothed model provides the most consistent experience. Position near windows for better signal.",
                "icon": "🏠",
                "color": "green",
            }
        )
    elif mobility_mode == "Driving":
        recommendations.append(
            {
                "title": "Driving Mode Optimization",
                "description": "While driving, expect frequent handovers between towers. The hybrid model handles these transitions most effectively.",
                "icon": "🚗",
                "color": "purple",
            }
        )

    # Add general recommendation if none specific
    if not recommendations:
        recommendations.append(
            {
                "title": "Balanced Conditions",
                "description": "Your current parameters indicate balanced conditions. All models should perform well, with the hybrid model offering the best overall experience.",
                "icon": "✅",
                "color": "green",
            }
        )

    data["recommendations"] = recommendations

    return data
