"""Silver layer database operations."""

from __future__ import annotations

import pandas as pd
from etl_studio.postgres.postgres import get_table, save_table


def get_table_from_bronze(table_name: str, preview: bool = False) -> pd.DataFrame:
    """Get a table from the bronze schema."""
    return get_table(table_name, "bronze", preview=preview)


def save_table_db(df: pd.DataFrame, table_name: str) -> None:
    """Save a DataFrame to the silver schema."""
    save_table(df, table_name, "silver")


def to_silver_db(df: pd.DataFrame, table_name: str) -> None:
    """Save a DataFrame to the silver schema."""
    save_table_db(df, table_name)
