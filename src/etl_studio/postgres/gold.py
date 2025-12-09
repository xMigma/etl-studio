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


def get_table_content_db(table_name: str, limit: Optional[int] = None) -> str:
    """Get content of a Gold table as CSV string."""
    clean_name = _clean_table_name(table_name)
    engine = get_engine()
    
    query = f'SELECT * FROM gold."{clean_name}"'
    if limit:
        query += f" LIMIT {limit}"
    
    df = pd.read_sql(query, engine)
    return df.to_csv(index=False)


def table_exists_db(table_name: str) -> bool:
    """Check if table exists in Gold schema."""
    clean_name = _clean_table_name(table_name)
    engine = get_engine()
    
    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = 'gold' AND table_name = :table_name
            )
        """),
            {"table_name": clean_name},
        )
        return result.scalar()


def delete_table_db(table_name: str) -> bool:
    """Delete a table from Gold schema."""
    if not table_exists_db(table_name):
        return False
    
    clean_name = _clean_table_name(table_name)
    engine = get_engine()
    
    with engine.connect() as conn:
        conn.execute(text(f'DROP TABLE gold."{clean_name}"'))
        conn.commit()
    
    return True