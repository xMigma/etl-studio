"""Silver layer cleaning helpers."""

from __future__ import annotations

from typing import Any

import pandas as pd
from etl_studio.postgres.silver import to_silver_db, get_preview_from_bronze, get_table_from_bronze
from etl_studio.api.schemas.silver import Operation

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


def dispatch_operations(
    table_name: str,
    selected_operations: list[Operation],
    preview: bool = True
) -> pd.DataFrame:
    """Get a table from bronze, apply cleaning operations, and optionally save to silver. """
    if preview:
        df = get_preview_from_bronze(table_name)
    else:
        df = get_table_from_bronze(table_name)
    
    for op in selected_operations:
        operation = op["operation"]
        params = op.get("params", {})
        df = apply_operation(df, operation, params)
    
    if not preview:
        to_silver_db(df, table_name)
    
    return df
