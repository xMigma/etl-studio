"""PostgreSQL database operations."""

from __future__ import annotations

import os
from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, inspect, text
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
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)

    if cleaned[0].isdigit():
        cleaned = f"t_{cleaned}"
    
    return cleaned


def get_table(table_name: str, schema: str, preview: bool = False) -> pd.DataFrame:
    """Get a table from a specific schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    
    if preview:
        query = text(f"SELECT * FROM {schema}.{cleaned_name} LIMIT 10")
    else:
        query = text(f"SELECT * FROM {schema}.{cleaned_name}")
    
    return pd.read_sql(query, engine)


def get_table_names(schema: str) -> list[str]:
    """Get all table names from a specific schema."""
    engine = get_engine()
    inspector = inspect(engine)
    return inspector.get_table_names(schema=schema)


def table_exists(table_name: str, schema: str) -> bool:
    """Check if a table exists in a specific schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    inspector = inspect(engine)
    
    return inspector.has_table(cleaned_name, schema=schema)


def get_table_columns(table_name: str, schema: str) -> list[dict]:
    """Get column information for a table."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    inspector = inspect(engine)
    
    if not inspector.has_table(cleaned_name, schema=schema):
        return []
    
    columns = inspector.get_columns(cleaned_name, schema=schema)
    return columns


def delete_table(table_name: str, schema: str) -> bool:
    """Delete a table from a specific schema."""
    if not table_exists(table_name, schema):
        return False
    
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {schema}.{cleaned_name}"))
        conn.commit()
    
    return True


def save_table(df: pd.DataFrame, table_name: str, schema: str) -> None:
    """Save a DataFrame to a specific schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))
        conn.commit()

    df_normalized = df.copy()
    df_normalized.columns = [col.lower().replace(" ", "_").replace("-", "_") for col in df.columns]
    
    df_normalized.to_sql(
        name=cleaned_name,
        con=engine,
        schema=schema,
        if_exists="replace",
        index=False
    )
