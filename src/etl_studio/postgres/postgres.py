"""PostgreSQL database operations."""

from __future__ import annotations

import os
from functools import lru_cache

from sqlalchemy import create_engine
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
