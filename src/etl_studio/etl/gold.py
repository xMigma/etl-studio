"""Gold layer integration helpers."""

from __future__ import annotations

import pandas as pd
from etl_studio.postgres.gold import get_table_db, get_table_names_db, delete_table_db, save_table_db, join_tables_db


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
    """Join two tables from specified sources."""
    result_table_name = f"{left_table}_{right_table}_joined"
    
    if preview:
        joined_df = join_tables_db(left_table, right_table, left_source, 
                                   right_source, left_key, right_key, join_type, preview=True)
    
    else:
        joined_df = join_tables_db(left_table, right_table, left_source, 
                                   right_source, left_key, right_key, join_type, preview=False)
        
        save_table_db(joined_df, result_table_name)

    return joined_df


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
    df = get_table_db(table_name, schema="gold", preview=preview)
    return df.to_csv(index=False)


def delete_table(table_name: str) -> bool:
    """Delete a specific table from the gold layer."""
    return delete_table_db(table_name)

