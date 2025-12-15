"""Machine learning helpers for ETL Studio."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import mlflow
import mlflow.sklearn
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
from etl_studio.config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME

# Configure MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

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

def apply_encoding(
    df: pd.DataFrame,
    encoding_config: dict[str, str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply encoding transformations to categorical features.
    
    Args:
        df: Original DataFrame
        encoding_config: Dict mapping column names to encoding types ('onehot' or 'label')
        
    Returns:
        Tuple of (encoded_df, encoders_dict) where encoders_dict contains the fitted encoders
    """
    df_encoded = df.copy()
    encoders = {}
    
    for column, encoding_type in encoding_config.items():
        if column not in df.columns:
            continue
            
        if encoding_type == "onehot":
            # One-Hot Encoding
            dummies = pd.get_dummies(df_encoded[column], prefix=column, drop_first=False)
            df_encoded = pd.concat([df_encoded.drop(columns=[column]), dummies], axis=1)
            encoders[column] = {"type": "onehot", "columns": dummies.columns.tolist()}
            
        elif encoding_type == "label":
            # Label Encoding
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
            encoders[column] = {"type": "label", "encoder": le, "classes": le.classes_.tolist()}
    
    return df_encoded, encoders

def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Get list of categorical columns in a DataFrame.
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        List of categorical column names
    """
    return df.select_dtypes(include=['object', 'category']).columns.tolist()

def train_model(
    df: pd.DataFrame,
    target_column: str,
    model_name: str,
    task_type: str,
    params: Optional[dict[str, Any]] = None,
    selected_features: Optional[list[str]] = None,
    test_size: float = 0.2,
    use_cross_validation: bool = True,
    run_name: Optional[str] = None,
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

    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_params({f"model_{k}": v for k, v in params.items()})    
    
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

    mlflow.log_param("train_samples", len(X_train))
    mlflow.log_param("test_samples", len(X_test))    
    
    # Train model
    model = model_class(**params)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    # Calculate metrics
    metrics = {}
    
    # TODO: check if metrics logging in mlflow works properly
    if task_type == "classification":
        # Classification metrics
        acc_train = accuracy_score(y_train, y_train_pred)
        acc_test = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        metrics.update({
            "accuracy_train": acc_train,
            "accuracy_test": acc_test,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": cm.tolist(),
        })
        
        # Log to MLflow
        mlflow.log_metric("accuracy_train", acc_train)
        mlflow.log_metric("accuracy_test", acc_test)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        
    else:
        # Regression metrics
        mae_train = mean_absolute_error(y_train, y_train_pred)
        mae_test = mean_absolute_error(y_test, y_pred)
        mse_train = mean_squared_error(y_train, y_train_pred)
        mse_test = mean_squared_error(y_test, y_pred)
        rmse_train = np.sqrt(mse_train)
        rmse_test = np.sqrt(mse_test)
        r2_train = r2_score(y_train, y_train_pred)
        r2_test = r2_score(y_test, y_pred)
        
        metrics.update({
            "mae_train": mae_train,
            "mae_test": mae_test,
            "mse_train": mse_train,
            "mse_test": mse_test,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "r2_train": r2_train,
            "r2_test": r2_test,
        })
        
        # Log to MLflow
        mlflow.log_metric("mae_train", mae_train)
        mlflow.log_metric("mae_test", mae_test)
        mlflow.log_metric("rmse_train", rmse_train)
        mlflow.log_metric("rmse_test", rmse_test)
        mlflow.log_metric("r2_train", r2_train)
        mlflow.log_metric("r2_test", r2_test)
    
    # Cross-validation
    if use_cross_validation:
        scoring = "accuracy" if task_type == "classification" else "r2"
        cv_scores = cross_val_score(model, X, y, cv=5, scoring=scoring)
        metrics["cv_scores"] = cv_scores.tolist()
        metrics["cv_mean"] = cv_scores.mean()
        metrics["cv_std"] = cv_scores.std()
        mlflow.log_metric("cv_mean", cv_scores.mean())
        mlflow.log_metric("cv_std", cv_scores.std())
    
    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)
        metrics["feature_importance"] = feature_importance.to_dict(orient="records")
        
        # Log feature importance as artifact
        importance_path = "feature_importance.csv"
        feature_importance.to_csv(importance_path, index=False)
        mlflow.log_artifact(importance_path)
        Path(importance_path).unlink()  

    # Log model to MLflow
    mlflow.sklearn.log_model(
        model,
        "model",
        registered_model_name=f"{model_name.replace(' ', '_')}_{task_type}"
    )

    # Save model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    # Get run info
    run_id = run.info.run_id
    metrics["mlflow_run_id"] = run_id
    metrics["mlflow_tracking_uri"] = MLFLOW_TRACKING_URI
    
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
        "mlflow_run_id": run_id,
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


def predict(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Run predictions using a stored model artifact.
    
    Args:
        df: DataFrame with features
        model_path: Path to pickled model
        
    Returns:
        DataFrame with predictions added
    """
    # Load model info
    with open(model_path, "rb") as f:
        model_info = pickle.load(f)
    
    model = model_info["model"]
    label_encoder = model_info.get("label_encoder")
    features = model_info["features"]
    
    # Select only the features used during training
    X = df[features]
    
    # Predict
    predictions = model.predict(X)
    
    # Decode labels if necessary
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(predictions)
    
    # Add predictions to dataframe
    result = df.copy()
    result["prediction"] = predictions
    
    # Add prediction probabilities for classification
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        for i, class_label in enumerate(
            label_encoder.classes_ if label_encoder else range(probas.shape[1])
        ):
            result[f"proba_{class_label}"] = probas[:, i]
    
    return result

def predict_from_mlflow(df: pd.DataFrame, run_id: str) -> pd.DataFrame:
    """Run predictions using a model from MLflow."""
    # Load model from MLflow
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    
    # Try to load label encoder
    label_encoder = None
    try:
        encoder_path = mlflow.artifacts.download_artifacts(
            run_id=run_id,
            artifact_path="label_encoder.pkl"
        )
        with open(encoder_path, "rb") as f:
            label_encoder = pickle.load(f)
    except:
        pass
    
    # Predict
    predictions = model.predict(df)
    
    # Decode labels if necessary
    if label_encoder is not None:
        predictions = label_encoder.inverse_transform(predictions)
    
    # Add predictions to dataframe
    result = df.copy()
    result["prediction"] = predictions
    
    # Add prediction probabilities for classification
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(df)
        for i, class_label in enumerate(
            label_encoder.classes_ if label_encoder else range(probas.shape[1])
        ):
            result[f"proba_{class_label}"] = probas[:, i]
    
    return result

def load_model_info(model_path: str) -> dict[str, Any]:
    """Load model metadata without the model object."""
    with open(model_path, "rb") as f:
        model_info = pickle.load(f)
    
    return {
        "model_name": model_info["model_name"],
        "task_type": model_info["task_type"],
        "target": model_info["target"],
        "features": model_info["features"],
        "params": model_info["params"],
    }