"""Silver layer database operations."""

from __future__ import annotations

import pandas as pd
from etl_studio.postgres.postgres import get_table_preview, get_table, save_table


def get_preview_from_bronze(table_name: str, limit: int = 5) -> pd.DataFrame:
    """Get a preview of a specific table from the bronze schema."""
    return get_table_preview(table_name, "bronze", limit)


def get_table_from_bronze(table_name: str) -> pd.DataFrame:
    """Get content of a specific table from the bronze schema."""
    return get_table(table_name, "bronze")


def to_silver_db(df: pd.DataFrame, table_name: str) -> None:
    """Save a DataFrame to the silver schema."""
    save_table(df, table_name, "silver")