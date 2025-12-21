"""Bronze layer database operations."""

from __future__ import annotations

import pandas as pd
from etl_studio.postgres.postgres import save_table, get_table_names, get_table, table_exists, delete_table


def save_table_db(df: pd.DataFrame, table_name: str) -> None:
    """Save a DataFrame to the bronze schema."""
    save_table(df, table_name, "bronze")


def get_table_db(table_name: str, preview: bool = False) -> pd.DataFrame:
    """Get a table from the bronze schema."""
    return get_table(table_name, "bronze", preview=preview)


def get_table_names_db() -> list[str]:
    """Get all table names from the bronze schema."""
    return get_table_names("bronze")


def table_exists_db(table_name: str) -> bool:
    """Check if a table exists in the bronze schema."""
    return table_exists(table_name, "bronze")


def delete_table_db(table_name: str) -> bool:
    """Delete a table from the bronze schema."""
    return delete_table(table_name, "bronze")


