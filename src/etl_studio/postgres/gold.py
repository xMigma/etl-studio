"""Gold layer database operations."""

from __future__ import annotations
from typing import Optional
import pandas as pd
from sqlalchemy import text
from etl_studio.postgres.postgres import get_engine


def _clean_table_name(table_name: str) -> str:
    """Adapt table name to valid SQL identifier."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in table_name.lower())

def save_table_db(table_name: str, data: list[dict]) -> None:
    """Save a table to the Gold schema."""
    clean_name = _clean_table_name(table_name)
    df = pd.DataFrame(data)
    
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS gold"))
        conn.commit()
    
    df.to_sql(
        name=clean_name,
        con=engine,
        schema="gold",
        if_exists="replace",
        index=False,
    )
