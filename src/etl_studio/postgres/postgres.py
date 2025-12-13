"""PostgreSQL database operations."""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


def generate_engine() -> Engine:
    """Create a SQLAlchemy engine from environment variables."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "etl_user")
    password = os.getenv("POSTGRES_PASSWORD", "etl_password")
    db = os.getenv("POSTGRES_DB", "etl_studio")

    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"

    engine = create_engine(url, pool_pre_ping=True, future=True)

    return engine


@lru_cache
def get_engine() -> Engine:
    """Return a singleton SQLAlchemy engine."""
    return generate_engine()


def clean_table_name(table_name: str) -> str:
    """Adapt the name of the table to a valid SQL identifier."""
    table_name = table_name.replace(".csv", "").lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)


def get_table(table_name: str, schema: str) -> pd.DataFrame:
    """Get a complete table from a specific schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    query = text(f"SELECT * FROM {schema}.{cleaned_name}")
    return pd.read_sql(query, engine)


def get_table_preview(table_name: str, schema: str, limit: int = 5) -> pd.DataFrame:
    """Get a preview of a table from a specific schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    query = text(f"SELECT * FROM {schema}.{cleaned_name} LIMIT :limit")
    return pd.read_sql(query, engine, params={"limit": limit})


def get_table_names(schema: str) -> list[str]:
    """Get all table names from a specific schema."""
    engine = get_engine()
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = :schema
            ORDER BY table_name
        """), {"schema": schema})
        
        return [row[0] for row in result]


def table_exists(table_name: str, schema: str) -> bool:
    """Check if a table exists in a specific schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    
    with engine.connect() as conn:
        result = conn.execute(text("""
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables 
                WHERE table_schema = :schema AND table_name = :table_name
            )
        """), {"schema": schema, "table_name": cleaned_name})
        
        return result.scalar()


def delete_table(table_name: str, schema: str) -> bool:
    """Delete a table from a specific schema."""
    if not table_exists(table_name, schema):
        return False
    
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE {schema}.{cleaned_name}"))
        conn.commit()
    
    return True


def save_table(df: pd.DataFrame, table_name: str, schema: str) -> None:
    """Save a DataFrame to a specific schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        conn.commit()
    
    df.to_sql(
        name=cleaned_name,
        con=engine,
        schema=schema,
        if_exists="replace",
        index=False
    )
