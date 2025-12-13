"""Bronze layer database operations."""

from __future__ import annotations

import pandas as pd
from etl_studio.postgres.postgres import save_table, get_table_names, get_table, table_exists, delete_table


def to_bronze_db(table_name: str, df: pd.DataFrame) -> None:
    """Write DataFrame to the bronze schema in PostgreSQL."""
    save_table(df, table_name, "bronze")

def get_table_names_db() -> list[str]:
    """Get all table names from the bronze schema."""
    return get_table_names("bronze")


def get_table_content_db(table_name: str) -> pd.DataFrame:
    """Get content of a specific table from the bronze schema."""
    return get_table(table_name, "bronze")


def table_exists_db(table_name: str) -> bool:
    """Check if a table exists in the bronze schema."""
    return table_exists(table_name, "bronze")


def delete_table_db(table_name: str) -> bool:
    """Delete a specific table from the bronze schema."""
    return delete_table(table_name, "bronze")
