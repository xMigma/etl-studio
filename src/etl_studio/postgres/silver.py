"""Silver layer database operations."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from etl_studio.postgres.postgres import get_engine, clean_table_name


def get_table_from_bronze(table_name: str) -> pd.DataFrame:
    """Get content of a specific table from the bronze schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    
    query = text("SELECT * FROM bronze.:table_name")
    return pd.read_sql(query, engine, params={"table_name": cleaned_name})


def save_table_to_silver(df: pd.DataFrame, table_name: str) -> None:
    """Save a DataFrame to the silver schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    
    df.to_sql(
        name=cleaned_name,
        con=engine,
        schema="silver",
        if_exists="replace",
        index=False
    )