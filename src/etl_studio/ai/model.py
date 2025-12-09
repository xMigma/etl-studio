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
            "max_depth": {"type": "slider", "min": 1, "max": 50, "default": 10},
            "min_samples_split": {"type": "slider", "min": 2, "max": 20, "default": 2},
            "min_samples_leaf": {"type": "slider", "min": 1, "max": 20, "default": 1},
        },
    },
    "Logistic Regression": {
        "class": LogisticRegression,
        "params": {
            "C": {"type": "slider", "min": 0.01, "max": 10.0, "default": 1.0, "step": 0.1},
            "max_iter": {"type": "slider", "min": 100, "max": 1000, "default": 100},
        },
    },
    "Gradient Boosting": {
        "class": GradientBoostingClassifier,
        "params": {
            "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100},
            "learning_rate": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01},
            "max_depth": {"type": "slider", "min": 1, "max": 20, "default": 3},
        },
    },
    "SVM": {
        "class": SVC,
        "params": {
            "C": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
            "kernel": {"type": "select", "options": ["linear", "rbf", "poly"], "default": "rbf"},
        },
    },
    "K-Nearest Neighbors": {
        "class": KNeighborsClassifier,
        "params": {
            "n_neighbors": {"type": "slider", "min": 1, "max": 50, "default": 5},
            "weights": {"type": "select", "options": ["uniform", "distance"], "default": "uniform"},
        },
    },
    "Decision Tree": {
        "class": DecisionTreeClassifier,
        "params": {
            "max_depth": {"type": "slider", "min": 1, "max": 50, "default": 5},
            "min_samples_split": {"type": "slider", "min": 2, "max": 20, "default": 2},
        },
    },
}

REGRESSION_MODELS = {
    "Random Forest": {
        "class": RandomForestRegressor,
        "params": {
            "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100},
            "max_depth": {"type": "slider", "min": 1, "max": 50, "default": 10},
            "min_samples_split": {"type": "slider", "min": 2, "max": 20, "default": 2},
        },
    },
    "Linear Regression": {
        "class": LinearRegression,
        "params": {},
    },
    "Ridge Regression": {
        "class": Ridge,
        "params": {
            "alpha": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
        },
    },
    "Lasso Regression": {
        "class": Lasso,
        "params": {
            "alpha": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
        },
    },
    "Gradient Boosting": {
        "class": GradientBoostingRegressor,
        "params": {
            "n_estimators": {"type": "slider", "min": 10, "max": 500, "default": 100},
            "learning_rate": {"type": "slider", "min": 0.01, "max": 1.0, "default": 0.1, "step": 0.01},
            "max_depth": {"type": "slider", "min": 1, "max": 20, "default": 3},
        },
    },
    "SVR": {
        "class": SVR,
        "params": {
            "C": {"type": "slider", "min": 0.1, "max": 10.0, "default": 1.0, "step": 0.1},
            "kernel": {"type": "select", "options": ["linear", "rbf", "poly"], "default": "rbf"},
        },
    },
    "K-Nearest Neighbors": {
        "class": KNeighborsRegressor,
        "params": {
            "n_neighbors": {"type": "slider", "min": 1, "max": 50, "default": 5},
            "weights": {"type": "select", "options": ["uniform", "distance"], "default": "uniform"},
        },
    },
    "Decision Tree": {
        "class": DecisionTreeRegressor,
        "params": {
            "max_depth": {"type": "slider", "min": 1, "max": 50, "default": 5},
            "min_samples_split": {"type": "slider", "min": 2, "max": 20, "default": 2},
        },
    },
}