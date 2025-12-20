"""Gold layer integration helpers."""

from __future__ import annotations

import pandas as pd
from etl_studio.postgres.gold import get_table_db, to_gold_db, get_table_names_db, delete_table_db


def join_tables(
    left_table: str,
    right_table: str,
    left_source: str,
    right_source: str,
    left_key: str,
    right_key: str,
    join_type: str = "inner",
    preview: bool = False
) -> pd.DataFrame:
    """Join two tables from specified sources. """

    left_df = get_table_db(left_table, left_source, preview=preview)
    right_df = get_table_db(right_table, right_source, preview=preview)
    
    if left_key not in left_df.columns:
        raise ValueError(f"Column '{left_key}' not found in {left_table}")
    if right_key not in right_df.columns:
        raise ValueError(f"Column '{right_key}' not found in {right_table}")
    
    result_df = pd.merge(
        left_df,
        right_df,
        left_on=left_key,
        right_on=right_key,
        how=join_type,
        suffixes=("_left", "_right")
    )
    
    if not preview:
        result_table_name = f"{left_table}_{right_table}_joined"
        to_gold_db(result_df, result_table_name)
    
    return result_df


def get_gold_tables_info() -> list[dict]:
    """Get all gold table names with their row counts."""
    table_names = get_table_names_db()
    result = []
    for table_name in table_names:
        df = get_table_db(table_name, "gold", preview=False)
        result.append({"name": table_name, "rows": len(df)})
    return result


def get_table(table_name: str, preview: bool = False) -> str:
    """Get content of a specific table as CSV string."""
    df = get_table_db(table_name, "gold", preview=preview)
    return df.to_csv(index=False)


def delete_table(table_name: str) -> bool:
    """Delete a specific table from the gold layer."""
    return delete_table_db(table_name)

