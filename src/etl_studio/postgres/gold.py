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

def get_table_names_db() -> list[dict]:
    """Get all table names from Gold schema with row counts."""
    engine = get_engine()
    
    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'gold'
            ORDER BY table_name
        """)
        )
        
        tables = []
        for row in result:
            table_name = row[0]
            count_result = conn.execute(
                text(f'SELECT COUNT(*) FROM gold."{table_name}"')
            )
            row_count = count_result.scalar()
            tables.append({"name": table_name, "rows": row_count})
        
        return tables
