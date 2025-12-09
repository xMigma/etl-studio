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


def get_available_models(task_type: str) -> dict:
    """Get available models for a task type."""
    if task_type == "classification":
        return CLASSIFICATION_MODELS
    elif task_type == "regression":
        return REGRESSION_MODELS
    else:
        raise ValueError(f"Unknown task type: {task_type}")


def train_model(
    df: pd.DataFrame,
    target_column: str,
    model_name: str,
    task_type: str,
    params: Optional[dict[str, Any]] = None,
    selected_features: Optional[list[str]] = None,
    test_size: float = 0.2,
    use_cross_validation: bool = True,
) -> dict[str, Any]:
    """Train a supervised model and return training metrics.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        model_name: Name of the model to use
        task_type: 'classification' or 'regression'
        params: Hyperparameters for the model
        selected_features: List of features to use (if None, use all)
        test_size: Proportion of data for testing
        use_cross_validation: Whether to use cross-validation
        
    Returns:
        Dictionary with training metrics and model info
    """
    if params is None:
        params = {}
    
    # Get model class
    models_dict = get_available_models(task_type)
    if model_name not in models_dict:
        raise ValueError(f"Unknown model: {model_name}")
    
    model_class = models_dict[model_name]["class"]
    
    # Prepare data
    if selected_features:
        X = df[selected_features]
    else:
        X = df.drop(columns=[target_column])
    
    y = df[target_column]
    
    # Handle categorical target for classification
    label_encoder = None
    if task_type == "classification" and y.dtype == "object":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Train model
    model = model_class(**params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    # Calculate metrics
    metrics = {}
    
    if task_type == "classification":
        metrics.update(
            {
                "accuracy_train": accuracy_score(y_train, y_train_pred),
                "accuracy_test": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            }
        )
    else:
        metrics.update(
            {
                "mae_train": mean_absolute_error(y_train, y_train_pred),
                "mae_test": mean_absolute_error(y_test, y_pred),
                "mse_train": mean_squared_error(y_train, y_train_pred),
                "mse_test": mean_squared_error(y_test, y_pred),
                "rmse_train": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                "rmse_test": np.sqrt(mean_squared_error(y_test, y_pred)),
                "r2_train": r2_score(y_train, y_train_pred),
                "r2_test": r2_score(y_test, y_pred),
            }
        )
    
    # Cross-validation
    if use_cross_validation:
        scoring = "accuracy" if task_type == "classification" else "r2"
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        metrics["cv_scores"] = cv_scores.tolist()
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()
    
    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        metrics["feature_importance"] = feature_importance.to_dict(orient="records")
    
    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{model_name.replace(' ', '_')}_{task_type}_{timestamp}.pkl"
    model_path = model_dir / model_filename
    
    model_info = {
        "model": model,
        "label_encoder": label_encoder,
        "features": X.columns.tolist(),
        "target": target_column,
        "task_type": task_type,
        "model_name": model_name,
        "params": params,
    }
    
    with open(model_path, "wb") as f:
        pickle.dump(model_info, f)
    
    metrics.update(
        {
            "model_path": str(model_path),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_used": X.columns.tolist(),
        }
    )
    
    return metrics