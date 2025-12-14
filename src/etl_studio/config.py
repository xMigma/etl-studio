"""Centralized configuration for ETL Studio."""

from __future__ import annotations

import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI_LOCAL", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = "etl_studio_models"