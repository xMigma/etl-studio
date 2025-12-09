"""Machine learning helpers for ETL Studio."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def train_model(
    df: pd.DataFrame,
    target_column: str,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, float]:
    """Train a supervised model and return training metrics.
    
    Args:
        df: DataFrame with features and target
        target_column: Name of target column
        params: Hyperparameters for the model
        
    Returns:
        Dictionary with training metrics
    """
    if params is None:
        params = {"n_estimators": 100, "max_depth": 10}
    
    # Separare features e target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entrenar modelo
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Guardar modelo
    model_path = Path("models") / f"model_{target_column}.pkl"
    model_path.parent.mkdir(exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    return {
        "accuracy": accuracy,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "model_path": str(model_path)
    }


def predict(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Run predictions using a stored model artifact.
    
    Args:
        df: DataFrame with features
        model_path: Path to pickled model
        
    Returns:
        DataFrame with predictions added
    """
    # Cargar modelo
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Predecir
    predictions = model.predict(df)
    
    # Agregar predicciones al dataframe
    result = df.copy()
    result["prediction"] = predictions
    
    return result