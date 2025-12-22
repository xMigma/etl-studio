"""Gold layer database operations."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text, inspect

from etl_studio.postgres.postgres import get_table as get_table, get_table_names, delete_table, save_table, get_engine, clean_table_name


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

def join_tables_db(
    left_table: str,
    right_table: str,
    left_schema: str,
    right_schema: str,
    left_key: str,
    right_key: str,
    join_type: str = "inner",
    preview: bool = False,
    output_table_name: str | None = None
) -> pd.DataFrame | None:
    """Execute a JOIN query. Uses Pandas only for preview."""

    engine = get_engine()

    left_table_clean = clean_table_name(left_table)
    right_table_clean = clean_table_name(right_table)
    join_clause = f"{join_type.upper()} JOIN"

    if preview:
        query = text(f"""
            SELECT *
            FROM (
                SELECT * FROM {left_schema}.{left_table_clean} LIMIT 5
            ) l
            {join_clause} (
                SELECT * FROM {right_schema}.{right_table_clean} LIMIT 5
            ) r
            ON l.{left_key}::text = r.{right_key}::text
        """)

        return pd.read_sql(query, engine)

    output_table_clean = clean_table_name(output_table_name)

    inspector = inspect(engine)

    left_cols = [c["name"] for c in inspector.get_columns(left_table_clean, schema=left_schema)]
    right_cols = [c["name"] for c in inspector.get_columns(right_table_clean, schema=right_schema)]

    right_cols_filtered = [c for c in right_cols if c not in left_cols]

    select_clause = ", ".join(
        [f"l.{c}" for c in left_cols] +
        [f"r.{c}" for c in right_cols_filtered]
    )

    query = text(f"""
        CREATE TABLE gold.{output_table_clean} AS
        SELECT {select_clause}
        FROM {left_schema}.{left_table_clean} l
        {join_clause} {right_schema}.{right_table_clean} r
        ON l.{left_key}::text = r.{right_key}::text
    """)

    with engine.begin() as conn:
        conn.execute(query)

    return None
