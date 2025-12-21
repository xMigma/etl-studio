"""Gold layer integration helpers."""

from __future__ import annotations

import pandas as pd
from etl_studio.postgres.gold import get_table_db, to_gold_db, get_table_names_db, delete_table_db, join_tables_db


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
        # Para preview: ejecutar JOIN en SQL pero SIN guardar tabla
        # Solo retornar el resultado temporal
        from etl_studio.postgres.postgres import get_engine
        from sqlalchemy import text
        
        engine = get_engine()
        
        # Mapear tipo de JOIN
        join_map = {
            "inner": "INNER JOIN",
            "left": "LEFT JOIN",
            "right": "RIGHT JOIN",
            "outer": "FULL OUTER JOIN"
        }
        join_clause = join_map.get(join_type, "INNER JOIN")
        
        # Query temporal sin crear tabla - solo SELECT con LIMIT
        if left_key == right_key:
            query = text(f"""
                SELECT * 
                FROM {left_source}.{left_table}
                {join_clause} {right_source}.{right_table}
                USING ({left_key})
                LIMIT 10
            """)
        else:
            query = text(f"""
                SELECT * 
                FROM {left_source}.{left_table} AS l
                {join_clause} {right_source}.{right_table} AS r
                ON l.{left_key} = r.{right_key}
                LIMIT 10
            """)
        
        return pd.read_sql(query, engine)
    
    else:
        # Para guardar: ejecutar JOIN completo y crear tabla en Gold
        join_tables_db(
            left_table, 
            right_table, 
            left_source, 
            right_source, 
            left_key, 
            right_key, 
            result_table_name,
            "gold",
            join_type
        )
        
        return get_table_db(result_table_name, "gold", preview=True)


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

