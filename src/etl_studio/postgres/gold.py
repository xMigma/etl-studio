"""Gold layer database operations."""

from __future__ import annotations

import pandas as pd

from etl_studio.postgres.postgres import get_table as get_table, get_table_names, delete_table, save_table, join_tables_sql


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

def join_tables_db( left_table: str, right_table: str, left_schema: str, right_schema: str, left_key: str,
    right_key: str, output_table: str, output_schema: str, join_type: str = "inner") -> None:
    """Join two tables using SQL."""
    join_tables_sql(left_table, right_table, left_schema, right_schema, left_key, 
    right_key, output_table, output_schema, join_type)
