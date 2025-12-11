"""Silver layer database operations."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from etl_studio.postgres.postgres import get_engine, clean_table_name


def get_preview_from_bronze(table_name: str, limit: int = 5) -> pd.DataFrame:
    """Get a preview of a specific table from the bronze schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    
    query = text("SELECT * FROM bronze.:table_name LIMIT :limit")
    return pd.read_sql(query, engine, params={"table_name": cleaned_name, "limit": limit})


def get_table_from_bronze(table_name: str) -> pd.DataFrame:
    """Get content of a specific table from the bronze schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    
    query = text("SELECT * FROM bronze.:table_name")
    return pd.read_sql(query, engine, params={"table_name": cleaned_name})


def to_silver_db(df: pd.DataFrame, table_name: str) -> None:
    """Save a DataFrame to the silver schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS silver"))
        conn.commit()
    
    df.to_sql(
        name=cleaned_name,
        con=engine,
        schema="silver",
        if_exists="replace",
        index=False
    )