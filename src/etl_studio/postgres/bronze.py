
"""Bronze layer ETL operations."""

from __future__ import annotations

import io

import pandas as pd
from sqlalchemy import text

from etl_studio.postgres.postgres import get_engine


def load_csv_to_bronze(filename: str, content: bytes) -> None:
    """Load CSV content to the bronze layer in PostgreSQL.
    
    Args:
        filename: Name of the uploaded file.
        content: Raw bytes content of the CSV file.
    """
    # Parse CSV content
    df = pd.read_csv(io.BytesIO(content))
    
    # Generate table name from filename
    table_name = filename.replace(".csv", "").lower()
    table_name = "".join(c if c.isalnum() or c == "_" else "_" for c in table_name)
    
    # Load to PostgreSQL
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS bronze"))
        conn.commit()
    
    # Write DataFrame to bronze schema
    df.to_sql(
        name=table_name,
        con=engine,
        schema="bronze",
        if_exists="replace",
        index=False
    )