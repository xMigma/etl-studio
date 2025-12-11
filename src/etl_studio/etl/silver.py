"""Silver layer cleaning helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd

def fillna(df: pd.DataFrame, column: str, value: Any) -> pd.DataFrame:
    """Fill null values in a specific column with the given value."""
    df[column] = df[column].fillna(value)
    return df

def drop_nulls(df: pd.DataFrame, column: str | None = None) -> pd.DataFrame:
    """Remove rows with null values. If column is specified, only check that column."""
    if column:
        return df.dropna(subset=[column])
    return df.dropna()

def drop_duplicates(df: pd.DataFrame, column: str | None = None) -> pd.DataFrame:
    """Remove duplicate rows. If column is specified, only check that column."""
    if column:
        return df.drop_duplicates(subset=[column])
    return df.drop_duplicates()

def lowercase(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Convert text in a column to lowercase."""
    if df[column].dtype == object:
        df[column] = df[column].str.lower()
    return df

def rename_column(df: pd.DataFrame, column: str, new_name: str) -> pd.DataFrame:
    """Rename a column."""
    return df.rename(columns={column: new_name})

def drop_column(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Delete a column from the DataFrame."""
    return df.drop(columns=[column])

def apply_operation(df: pd.DataFrame, operation: str, params: dict[str, Any]) -> pd.DataFrame:
    """Apply a single operation to the DataFrame based on operation key."""
    
    if operation == "fillna":
        return fillna(df, params["column"], params["value"])
    
    elif operation == "drop_nulls":
        return drop_nulls(df, params.get("column"))
    
    elif operation == "drop_duplicates":
        return drop_duplicates(df, params.get("column"))
    
    elif operation == "lowercase":
        return lowercase(df, params["column"])
    
    elif operation == "rename_column":
        return rename_column(df, params["column"], params["new_name"])
    
    elif operation == "drop_column":
        return drop_column(df, params["column"])
    
    else:
        raise ValueError(f"Unknown operation: {operation}")


def dispatch_operations(df: pd.DataFrame, selected_operations: list[dict[str, Any]]) -> pd.DataFrame:
    """Receive and apply the operations"""
    for op in selected_operations:
        operation = op["operation"]
        params = op.get("params", {})
        df = apply_operation(df, operation, params)
    
    return df

