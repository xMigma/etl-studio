"""Bronze layer ETL operations."""

from __future__ import annotations

import io

import pandas as pd

from etl_studio.postgres.bronze import get_table_db, save_table_db, get_table_names_db, delete_table_db


def load_csv_to_bronze(filename: str, content: bytes) -> None:
    """Parse CSV and load to bronze layer."""
    df = pd.read_csv(io.BytesIO(content))
    save_table_db(df, filename)

def get_bronze_tables_info() -> list[dict]:
    """Get all bronze table names with their row counts."""
    table_names = get_table_names_db()
    result = []
    for table_name in table_names:
        df = get_table_db(table_name)
        result.append({"name": table_name, "rows": len(df)})
    return result

def get_table(table_name: str, preview: bool = False) -> str:
    """Get content of a specific table as CSV string."""
    df = get_table_db(table_name, preview=preview)
    return df.to_csv(index=False)

def delete_table(table_name: str) -> bool:
    """Delete a specific table from the bronze layer."""
    return delete_table_db(table_name)