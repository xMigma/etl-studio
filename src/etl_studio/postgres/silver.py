"""Silver layer database operations."""

from __future__ import annotations

import pandas as pd
from etl_studio.postgres.postgres import delete_table, get_table, save_table, get_table_names


def get_table_db(table_name: str, schema: str, preview: bool = False) -> pd.DataFrame:
    """Get a table from a specific schema (silver or bronze)."""
    return get_table(table_name, schema, preview=preview)

def get_table_names_db() -> list[str]:
    """Get all table names from the silver schema."""
    return get_table_names("silver")

def delete_table_db(table_name: str) -> bool:
    """Delete a table from the silver schema."""
    return delete_table(table_name, "silver")

def save_table_db(df: pd.DataFrame, table_name: str) -> None:
    """Save a DataFrame to the silver schema."""
    save_table(df, table_name, "silver")
