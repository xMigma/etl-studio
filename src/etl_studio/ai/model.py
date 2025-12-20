"""Machine learning helpers for ETL Studio with MLflow integration."""

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


# Model configurations (same as before)
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
    """Apply encoding transformations to categorical features."""
    df_encoded = df.copy()
    encoders = {}
    
    for column, encoding_type in encoding_config.items():
        if column not in df.columns:
            continue
            
        if encoding_type == "onehot":
            dummies = pd.get_dummies(df_encoded[column], prefix=column, drop_first=False)
            df_encoded = pd.concat([df_encoded.drop(columns=[column]), dummies], axis=1)
            encoders[column] = {"type": "onehot", "columns": dummies.columns.tolist()}
            
        elif encoding_type == "label":
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column].astype(str))
            encoders[column] = {"type": "label", "encoder": le, "classes": le.classes_.tolist()}
    
    return df_encoded, encoders


def apply_feature_encoders(df: pd.DataFrame, encoders: dict[str, Any]) -> pd.DataFrame:
    """Apply previously fitted encoders to a DataFrame for predictions."""
    df_encoded = df.copy()
    
    for column, encoder_info in encoders.items():
        if column not in df.columns:
            continue
            
        if encoder_info["type"] == "onehot":
            # Apply one-hot encoding
            dummies = pd.get_dummies(df_encoded[column], prefix=column, drop_first=False)
            df_encoded = pd.concat([df_encoded.drop(columns=[column]), dummies], axis=1)
            
            # Ensure all columns from training are present
            for col in encoder_info["columns"]:
                if col not in df_encoded.columns:
                    df_encoded[col] = 0
            
            # Remove any extra columns not seen during training
            extra_cols = [col for col in df_encoded.columns if col.startswith(f"{column}_") and col not in encoder_info["columns"]]
            if extra_cols:
                df_encoded = df_encoded.drop(columns=extra_cols)
                
        elif encoder_info["type"] == "label":
            # Apply label encoding
            le = encoder_info["encoder"]
            # Handle unseen labels by encoding them as -1 or the most common class
            df_encoded[column] = df_encoded[column].astype(str).apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else 0
            )
    
    return df_encoded


def get_categorical_columns(df: pd.DataFrame) -> list[str]:
    """Get list of categorical columns in a DataFrame."""
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
    feature_encoders: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Train a supervised model and log to MLflow."""
    if params is None:
        params = {}
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("task_type", task_type)
        mlflow.log_param("target_column", target_column)
        mlflow.log_param("test_size", test_size)
        mlflow.log_param("use_cross_validation", use_cross_validation)
        mlflow.log_params({f"model_{k}": v for k, v in params.items()})
        
        # Get model class
        models_dict = get_available_models(task_type)
        if model_name not in models_dict:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_class = models_dict[model_name]["class"]
        
        # Prepare data
        if selected_features:
            X = df[selected_features]
            mlflow.log_param("n_features", len(selected_features))
            mlflow.log_param("features", ",".join(selected_features))
        else:
            X = df.drop(columns=[target_column])
            mlflow.log_param("n_features", len(X.columns))
        
        y = df[target_column]
        
        # Handle categorical target for classification
        label_encoder = None
        if task_type == "classification" and y.dtype == "object":
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            mlflow.log_param("target_classes", ",".join(label_encoder.classes_))
        
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
        
        # Calculate and log metrics
        metrics = {}
        
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
            feature_importance = pd.DataFrame({
                "feature": X.columns,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            metrics["feature_importance"] = feature_importance.to_dict(orient="records")
            
            # Log feature importance as artifact
            importance_path = "feature_importance.csv"
            feature_importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(importance_path)
            Path(importance_path).unlink()  # Clean up
        
        # Log model to MLflow with signature
        from mlflow.models.signature import infer_signature
        signature = infer_signature(X_train, y_train)
        
        mlflow.sklearn.log_model(
            model,
            "model",
            signature=signature,
            registered_model_name=f"{model_name.replace(' ', '_')}_{task_type}"
        )
        
        # Save additional artifacts (label encoder if exists)
        if label_encoder is not None:
            encoder_path = "label_encoder.pkl"
            with open(encoder_path, "wb") as f:
                pickle.dump(label_encoder, f)
            mlflow.log_artifact(encoder_path)
            Path(encoder_path).unlink()
        
        # Get run info
        run_id = run.info.run_id
        metrics["mlflow_run_id"] = run_id
        metrics["mlflow_tracking_uri"] = MLFLOW_TRACKING_URI
        
        # Also save locally for backward compatibility
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{model_name.replace(' ', '_')}_{task_type}_{timestamp}.pkl"
        model_path = model_dir / model_filename
        
        model_info = {
            "model": model,
            "label_encoder": label_encoder,
            "feature_encoders": feature_encoders,
            "features": X.columns.tolist(),
            "target": target_column,
            "task_type": task_type,
            "model_name": model_name,
            "params": params,
            "mlflow_run_id": run_id,
        }
        
        with open(model_path, "wb") as f:
            pickle.dump(model_info, f)
        
        metrics.update({
            "model_path": str(model_path),
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features_used": X.columns.tolist(),
        })
        
        return metrics


def predict(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Run predictions using a stored model artifact."""
    # Load model info
    with open(model_path, "rb") as f:
        model_info = pickle.load(f)
    
    model = model_info["model"]
    label_encoder = model_info.get("label_encoder")
    feature_encoders = model_info.get("feature_encoders")
    features = model_info["features"]
    
    # Apply feature encoders if they exist
    df_processed = df.copy()
    if feature_encoders:
        df_processed = apply_feature_encoders(df_processed, feature_encoders)
    else:
        # Check if features are missing - indicates old model without encoders
        missing_features = [f for f in features if f not in df_processed.columns]
        if missing_features:
            raise ValueError(
                f"Este modelo fue entrenado con una versión antigua y no tiene encoders guardados. "
                f"Por favor, re-entrena el modelo para usar la funcionalidad de predicción. "
                f"Features faltantes: {missing_features[:5]}..."
            )
    
    # Select only the features used during training
    X = df_processed[features]
    
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
    
    # Try to load label encoder if exists
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
        "mlflow_run_id": model_info.get("mlflow_run_id"),
    }


def get_mlflow_runs(limit: int = 50) -> pd.DataFrame:
    """Get recent MLflow runs for the experiment."""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    
    if experiment is None:
        return pd.DataFrame()
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=limit
    )
    
    runs_data = []
    for run in runs:
        runs_data.append({
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName", ""),
            "model_name": run.data.params.get("model_name", ""),
            "task_type": run.data.params.get("task_type", ""),
            "accuracy_test": run.data.metrics.get("accuracy_test"),
            "r2_test": run.data.metrics.get("r2_test"),
            "mae_test": run.data.metrics.get("mae_test"),
            "start_time": pd.Timestamp(run.info.start_time, unit='ms'),
            "status": run.info.status,
        })
    
    return pd.DataFrame(runs_data)


def get_registered_models() -> list[dict]:
    """Get all registered models from MLflow Model Registry."""
    client = mlflow.tracking.MlflowClient()
    
    try:
        registered_models = client.search_registered_models()
        models = []
        
        for rm in registered_models:
            latest_versions = client.get_latest_versions(rm.name)
            for version in latest_versions:
                models.append({
                    "name": rm.name,
                    "version": version.version,
                    "stage": version.current_stage,
                    "run_id": version.run_id,
                    "creation_timestamp": pd.Timestamp(version.creation_timestamp, unit='ms'),
                })
        
        return models
    except:
        return []