"""Gold layer database operations."""

from __future__ import annotations

import pandas as pd

from etl_studio.postgres.postgres import get_table as get_table, get_table_names, delete_table, save_table


def get_table_db(table_name: str, schema: str, preview: bool = False) -> pd.DataFrame:
    """Get a table from a specific schema (silver or gold)."""
    return get_table(table_name, schema, preview=preview)


def to_gold_db(df: pd.DataFrame, table_name: str) -> None:
    """Save a DataFrame to the gold schema."""
    save_table(df, table_name, "gold")


def get_table_names_db() -> list[str]:
    """Get all table names from the gold schema."""
    return get_table_names("gold")

def delete_table_db(table_name: str) -> bool:
    """Delete a table from the gold schema."""
    return delete_table(table_name, "gold")
