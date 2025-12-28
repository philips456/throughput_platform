from django.urls import path
from . import views

urlpatterns = [
    # Authentication routes
    path("", views.login_view, name="login"),
    path("signup/", views.signup_view, name="signup"),
    path("logout/", views.logout_view, name="logout"),
    # Main pages
    path("dashboard/", views.dashboard, name="dashboard"),
    path(
        "predict/", views.visuals_page, name="predict_page"
    ),  # This is still here for backward compatibility
    path(
        "predictor/", views.predict_page, name="predictor_page"
    ),  # Direct access to predictor page
    path(
        "visuals/", views.visuals_page, name="visuals_page"
    ),  # New direct access to visuals page
    path("drift/", views.drift_page, name="drift_page"),
    path("map/", views.map_page, name="map_page"),
    path("sdg/", views.sdg_page, name="sdg_page"),
    # MLOps and metrics pages
    path("mlops/", views.mlops, name="mlops_page"),
    path("metrics/", views.system_metrics, name="system_metrics"),
    path("metrics/cpu/", views.cpu_metrics, name="cpu_metrics"),
    path("metrics/sessions/", views.sessions_metrics, name="sessions_metrics"),
    path("metrics/users/", views.users_metrics, name="users_metrics"),
    path("models/", views.models_page, name="models_page"),
    # New MLflow and model pages
    path("mlflow/metrics/", views.mlflow_metrics, name="mlflow_metrics"),
    path("models/raw/", views.raw_model, name="raw_model"),
    path("models/smoothed/", views.smoothed_model, name="smoothed_model"),
    path("models/hybrid/", views.hybrid_model, name="hybrid_model"),
    path("models/compare/", views.compare_models, name="compare_models"),
    path("metrics/load-history/", views.load_history, name="load_history"),
    # API endpoints
    path("api/get_antennas/", views.get_antennas, name="get_antennas"),
    path("api/map_prediction/", views.map_prediction, name="map_prediction"),
    path(
        "api/map_prediction_api/", views.map_prediction_api, name="map_prediction_api"
    ),
    path("api/predict/", views.predict_api, name="predict_api"),
    path("api/detect_drift/", views.detect_drift, name="detect_drift"),
    path(
        "api/predict_throughput/", views.predict_throughput, name="predict_throughput"
    ),
    # New API endpoints for visualization
    path("api/scenario_data/", views.scenario_data, name="scenario_data"),
    path(
        "api/custom_scenario_data/",
        views.custom_scenario_data,
        name="custom_scenario_data",
    ),
    # AI Assistant Endpoint
]
