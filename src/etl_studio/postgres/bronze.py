"""Bronze layer database operations."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from etl_studio.postgres.postgres import get_engine


def to_bronze_db(table_name: str, df: pd.DataFrame) -> None:
    """Write DataFrame to the bronze schema in PostgreSQL.
    
    Args:
        table_name: Sanitized table name.
        df: DataFrame to write.
    """
    engine = get_engine()
    with engine.connect() as conn:
        conn.execute(text("CREATE SCHEMA IF NOT EXISTS bronze"))
        conn.commit()
    
    df.to_sql(
        name=table_name,
        con=engine,
        schema="bronze",
        if_exists="replace",
        index=False
    )