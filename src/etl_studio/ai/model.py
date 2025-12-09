"""Machine learning helpers for ETL Studio."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.preprocessing import LabelEncoder


# Configuración de modelos disponibles
CLASSIFICATION_MODELS = {
    "Random Forest": {
        "class": RandomForestClassifier,
        "params": {
            "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100},
            "min_samples_leaf": {"type": "slider", "min": 1, "max": 20, "default": 1},
        },
    },
    "Logistic Regression": {
        "class": LogisticRegression,
        "params": {
        },
    },
    "Gradient Boosting": {
        "class": GradientBoostingClassifier,
        "params": {
        },
    },
    "SVM": {
        "class": SVC,
        "params": {
        },
    },
    "K-Nearest Neighbors": {
        "class": KNeighborsClassifier,
        "params": {
        },
    },
}