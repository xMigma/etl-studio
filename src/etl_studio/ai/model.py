"""Machine learning helpers for ETL Studio."""

from __future__ import annotations

from typing import Any

import pandas as pd


def train_model(
    df: pd.DataFrame,
    target_column: str,
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Train a supervised model and return training metrics."""

    # TODO: Implement feature engineering, training loop, and evaluation.
    print(
        f"[Model] Training placeholder on df shape={df.shape}, target={target_column}, params={params}"
    )
    return {"accuracy": 0.0, "loss": 0.0}


def predict(df: pd.DataFrame, model_path: str) -> pd.DataFrame:
    """Run predictions using a stored model artifact."""

    # TODO: Load serialized model from `model_path` and generate predictions.
    print(f"[Model] Predict placeholder using model_path={model_path}")
    predictions = df.copy()
    predictions["prediction"] = 0.0
    return predictions
