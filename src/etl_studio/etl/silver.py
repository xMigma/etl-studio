"""Silver layer cleaning helpers."""

from __future__ import annotations

from typing import Any, Callable

import pandas as pd
from etl_studio.postgres.silver import to_silver_db, get_preview_from_bronze, get_table_from_bronze

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

def groupby_agg(df: pd.DataFrame, group_columns: list[str], aggregations: dict[str,str]) -> pd.DataFrame:
    """Group by a list of columns with given aggregations"""
    result = df.groupby(group_columns).agg(aggregations)

    new_columns = {}
    for col, func in aggregations.items():
        new_columns[col] = f"{col}_{func}"

    return result.rename(columns=new_columns).reset_index()

OperationFn = Callable[[pd.DataFrame, dict[str, Any]], pd.DataFrame]

OP_FUNCS: dict[str, OperationFn] = {
    "fillna": lambda df, p: fillna(df, p["column"], p["value"]),
    "drop_nulls": lambda df, p: drop_nulls(df, p.get("column")),
    "drop_duplicates": lambda df, p: drop_duplicates(df, p.get("column")),
    "lowercase": lambda df, p: lowercase(df, p["column"]),
    "rename_column": lambda df, p: rename_column(df, p["column"], p["new_name"]),
    "drop_column": lambda df, p: drop_column(df, p["column"]),
    "groupby": lambda df, p: groupby_agg(df, p["group_columns"], p["aggregations"]),
}

def apply_operation(df: pd.DataFrame, operation: str, params: dict[str, Any]) -> pd.DataFrame:
    """Apply a single operation to the DataFrame based on operation key."""
    try:
        op = OP_FUNCS[operation]
    except KeyError:
        raise ValueError(f"Unknown operation: {operation}")

    return op(df, params)

def dispatch_operations(
    table_name: str,
    operations: list[dict[str, Any]],
    preview: bool = True
) -> pd.DataFrame:
    """Get a table from bronze, apply cleaning operations, and optionally save to silver."""
    df = get_preview_from_bronze(table_name) if preview else get_table_from_bronze(table_name)

    for op in operations:
        operation = op["operation"]
        params = op.get("params", {})
        df = apply_operation(df, operation, params)

    if not preview:
        to_silver_db(df, table_name)

    return df
