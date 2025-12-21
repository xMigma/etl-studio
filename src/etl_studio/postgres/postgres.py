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
    return "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)


def get_table(table_name: str, schema: str, preview: bool = False) -> pd.DataFrame:
    """Get a table from a specific schema."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    
    if preview:
        query = text(f"SELECT * FROM {schema}.{cleaned_name} LIMIT 10")
    else:
        query = text(f"SELECT * FROM {schema}.{cleaned_name}")
    
    return pd.read_sql(query, engine)


def get_table_preview(table_name: str, schema: str, limit: int = 5) -> pd.DataFrame:
    """Get a preview of a table from a specific schema (legacy, use get_table with preview=True)."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    query = text(f"SELECT * FROM {schema}.{cleaned_name} LIMIT :limit")
    return pd.read_sql(query, engine, params={"limit": limit})


def get_table_names(schema: str) -> list[str]:
    """Get all table names from a specific schema using SQLAlchemy Inspector."""
    engine = get_engine()
    inspector = inspect(engine)
    return inspector.get_table_names(schema=schema)


def table_exists(table_name: str, schema: str) -> bool:
    """Check if a table exists in a specific schema using SQLAlchemy Inspector."""
    cleaned_name = clean_table_name(table_name)
    engine = get_engine()
    inspector = inspect(engine)
    
    return inspector.has_table(cleaned_name, schema=schema)


def get_table_columns(table_name: str, schema: str) -> list[dict]:
    """Get column information for a table using SQLAlchemy Inspector."""
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


def join_tables_sql(left_table: str, right_table: str, left_schema: str, right_schema: str, left_key: str, 
    right_key: str, output_table: str, output_schema: str, join_type: str = "inner") -> None:
    """Executes the Join SQL query for big tables"""
    join_map = {
        "inner": "INNER JOIN",
        "left": "LEFT JOIN",
        "right": "RIGHT JOIN",
        "outer": "FULL OUTER JOIN",
    }
    join_clause = join_map.get(join_type, "INNER JOIN")
    
    engine = get_engine()
    
    with engine.connect() as conn:
        conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {output_schema}"))
        conn.execute(text(f"DROP TABLE IF EXISTS {output_schema}.{output_table}"))
        
        # Use USING clause if both keys are the same to avoid duplicate columns
        if left_key == right_key:
            query = text(f"""
                CREATE TABLE {output_schema}.{output_table} AS
                SELECT *
                FROM {left_schema}.{left_table}
                {join_clause} {right_schema}.{right_table}
                USING ({left_key})
            """)
        else:
            query = text(f"""
                CREATE TABLE {output_schema}.{output_table} AS
                SELECT *
                FROM {left_schema}.{left_table} AS l
                {join_clause} {right_schema}.{right_table} AS r
                ON l.{left_key} = r.{right_key}
            """)
        
        conn.execute(query)
        conn.commit()


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
