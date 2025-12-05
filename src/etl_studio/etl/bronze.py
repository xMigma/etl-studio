"""Bronze layer ETL operations."""

from __future__ import annotations

import io

import pandas as pd

from etl_studio.postgres.bronze import to_bronze_db, get_table_names_db


def load_csv_to_bronze(filename: str, content: bytes) -> None:
    """Parse CSV and load to bronze layer."""
    df = pd.read_csv(io.BytesIO(content))
    table_name = _clean_table_name(filename)
    to_bronze_db(table_name, df)

def _clean_table_name(filename: str) -> str:
    """Convert filename to valid SQL table name."""
    table_name = filename.replace(".csv", "").lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)

def get_bronze_table_names() -> list[dict]:
    """Get all bronze table names."""
    table_names = get_table_names_db()
    return [{"name": name} for name in table_names]