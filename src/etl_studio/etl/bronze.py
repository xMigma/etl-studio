"""Bronze layer ETL operations."""

from __future__ import annotations

import io
import io
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from etl_studio.postgres.bronze import get_table_content_db, to_bronze_db, get_table_names_db, delete_table_db


def load_csv_to_bronze(filename: str, content: bytes) -> None:
    """Parse CSV and load to bronze layer."""
    df = pd.read_csv(io.BytesIO(content))
    to_bronze_db(filename, df)

def get_bronze_table_names() -> list[dict]:
    """Get all bronze table names."""
    table_names = get_table_names_db()
    return [{"name": name, "rows": get_table_content_db(name).shape[0]} for name in table_names]

def get_table_content(table_name: str, limit: Optional[int] = None) -> str:
    """Get content of a specific table as CSV string."""
    df = get_table_content_db(table_name)
    if limit:
        df = df.head(limit)
    return df.to_csv(index=False)

def delete_table(table_name: str) -> bool:
    """Delete a specific table from the bronze layer."""
    return delete_table_db(table_name)