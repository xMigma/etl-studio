"""Gold layer database operations."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from etl_studio.postgres.postgres import get_table as get_table, get_table_names, delete_table, save_table, join_tables_sql, get_engine


def get_table_db(table_name: str, schema: str, preview: bool = False) -> pd.DataFrame:
    """Get a table from a specific schema (silver or gold)."""
    return get_table(table_name, schema, preview=preview)

def save_table_db(df: pd.DataFrame, table_name: str) -> None:
    """Save a DataFrame to the gold schema."""
    save_table(df, table_name, "gold")

def get_table_names_db() -> list[str]:
    """Get all table names from the gold schema."""
    return get_table_names("gold")

def delete_table_db(table_name: str) -> bool:
    """Delete a table from the gold schema."""
    return delete_table(table_name, "gold")

def join_tables_db(left_table: str, right_table: str, left_schema: str, right_schema: str, 
                           left_key: str, right_key: str, join_type: str = "inner", preview: bool = False) -> pd.DataFrame:
    """Execute a JOIN query with SQL."""
    join_clause = f"{join_type.upper()} JOIN"
    limit_clause = "LIMIT 10" if preview else ""
    
    engine = get_engine()
    
    if left_key == right_key:
        query = text(f"""
            SELECT * 
            FROM {left_schema}.{left_table}
            {join_clause} {right_schema}.{right_table}
            USING ({left_key})
            {limit_clause}
        """)
    else:
        query = text(f"""
            SELECT * 
            FROM {left_schema}.{left_table} AS l
            {join_clause} {right_schema}.{right_table} AS r
            ON l.{left_key} = r.{right_key}
            {limit_clause}
        """)
    
    return pd.read_sql(query, engine)
