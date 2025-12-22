"""Gold layer database operations."""

from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from etl_studio.postgres.postgres import get_table as get_table, get_table_names, delete_table, save_table, get_engine, clean_table_name


def get_table_db(table_name: str, schema: str, preview: bool = False) -> pd.DataFrame:
    """Get a table from a specific schema (silver or gold)."""
    return get_table(table_name, schema, preview=preview)

def save_table_db(df: pd.DataFrame, table_name: str) -> None:
    """Save a DataFrame to the gold schema."""
    save_table(df, table_name, "gold")

def get_table_names_db() -> list[str]:
    """Get all table names from the gold schema."""
    return get_table_names("gold")

def delete_table_db(table_name: str) -> bool:
    """Delete a table from the gold schema."""
    return delete_table(table_name, "gold")

def join_tables_db(left_table: str, right_table: str, left_schema: str, right_schema: str, 
                           left_key: str, right_key: str, join_type: str = "inner", preview: bool = False) -> pd.DataFrame:
    """Execute a JOIN query with SQL."""
    from sqlalchemy import inspect as sql_inspect
    
    left_table_clean = clean_table_name(left_table)
    right_table_clean = clean_table_name(right_table)
    
    join_clause = f"{join_type.upper()} JOIN"
    limit_clause = "LIMIT 10" if preview else ""
    
    engine = get_engine()
    
    try:
        if left_key == right_key:
            query = text(f"""
                SELECT * 
                FROM {left_schema}.{left_table_clean}
                {join_clause} {right_schema}.{right_table_clean}
                USING ({left_key})
                {limit_clause}
            """)
        else:
            inspector = sql_inspect(engine)
            
            left_cols = [col['name'] for col in inspector.get_columns(left_table_clean, schema=left_schema)]
            right_cols = [col['name'] for col in inspector.get_columns(right_table_clean, schema=right_schema)]
            
            left_select = ", ".join([f"l.{col}" for col in left_cols])
            right_select = ", ".join([f"r.{col}" for col in right_cols if col not in left_cols])
            
            select_clause = left_select
            if right_select:
                select_clause += ", " + right_select
            
            query = text(f"""
                SELECT {select_clause}
                FROM {left_schema}.{left_table_clean} l
                {join_clause} {right_schema}.{right_table_clean} r
                ON l.{left_key} = r.{right_key}
                {limit_clause}
            """)
        
        result = pd.read_sql(query, engine)
        
        if result is None:
            raise ValueError("pd.read_sql returned None")
        
        return result
        
    except Exception as e:
        raise ValueError(f"Error al realizar el join: {str(e)}")

