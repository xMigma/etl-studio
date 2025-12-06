"""Bronze layer database operations."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from etl_studio.postgres.postgres import get_engine


def _clean_table_name(table_name: str) -> str:
    """Adapt the name of the table to a valid SQL identifier."""
    table_name = table_name.replace(".csv", "").lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)

def to_bronze_db(table_name: str, df: pd.DataFrame) -> None:
    """Write DataFrame to the bronze schema in PostgreSQL."""
    clean_table_name = _clean_table_name(table_name)
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS bronze"))
        conn.commit()
    
    df.to_sql(
        name=clean_table_name,
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
    clean_table_name = _clean_table_name(table_name)
    engine = get_engine()
    
    query = "SELECT * FROM bronze." + clean_table_name
    return pd.read_sql(query, engine)

def table_exists_db(table_name: str) -> bool:
    """Check if a table exists in the bronze schema."""
    clean_table_name = _clean_table_name(table_name)
    engine = get_engine()
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'bronze' AND table_name = :table_name
            )
        """), {"table_name": clean_table_name})
        
        return result.scalar()

def delete_table_db(table_name: str) -> bool:
    """Delete a specific table from the bronze schema."""
    if not table_exists_db(table_name):
        return False
    
    clean_table_name = _clean_table_name(table_name)
    engine = get_engine()
    
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE bronze." + clean_table_name))
        conn.commit()
    
    return True