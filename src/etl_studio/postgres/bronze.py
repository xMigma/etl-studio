"""Bronze layer database operations."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from etl_studio.postgres.postgres import get_engine


def to_bronze_db(table_name: str, df: pd.DataFrame) -> None:
    """Write DataFrame to the bronze schema in PostgreSQL."""
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS bronze"))
        conn.commit()
    
    df.to_sql(
        name=table_name,
        con=engine,
        schema="bronze",
        if_exists="replace",
        index=False
    )

def get_table_names_db() -> list[str]:
    """Get all table names from the bronze schema."""
    engine = get_engine()
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'bronze'
            ORDER BY table_name
        """))
        
        return [row[0] for row in result]

def get_table_content_db(table_name: str) -> pd.DataFrame:
    """Get content of a specific table from the bronze schema."""
    engine = get_engine()
    
    query = f"SELECT * FROM bronze.{table_name}"
    return pd.read_sql(query, engine)