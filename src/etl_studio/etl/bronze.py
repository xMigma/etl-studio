"""Bronze layer ETL operations."""

from __future__ import annotations

import io

import pandas as pd

from etl_studio.postgres.bronze import to_bronze_db


def load_csv_to_bronze(filename: str, content: bytes) -> None:
    """Parse CSV and load to bronze layer.
    
    Args:
        filename: Name of the uploaded file.
        content: Raw bytes content of the CSV file.
    """
    # Parse CSV content
    df = pd.read_csv(io.BytesIO(content))
    
    # Generate table name from filename
    table_name = _sanitize_table_name(filename)
    
    # Load to database
    to_bronze_db(table_name, df)


def _sanitize_table_name(filename: str) -> str:
    """Convert filename to valid SQL table name."""
    table_name = filename.replace(".csv", "").lower()
    return "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)