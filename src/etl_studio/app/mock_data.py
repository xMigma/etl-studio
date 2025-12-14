"""Mock data for development and testing when API is unavailable."""

from pathlib import Path
from typing import Any, Optional

import json
import pandas as pd

from etl_studio.etl.silver import groupby_agg

# Ruta a los CSVs de bronze para fallback
BRONZE_PATH = Path(__file__).parent.parent.parent.parent / "data" / "bronze"

MOCK_TABLES = [
    {"name": "customers", "rows": 5},
    {"name": "orders", "rows": 7},
    {"name": "products", "rows": 5},
]

# Reglas mock para cuando la API no está disponible
MOCK_RULES = {
    "fillna": {"name": "FillNA", "description": "Rellenar valores nulos", "requires_value": True},
    "trim": {"name": "Trim", "description": "Eliminar espacios en blanco", "requires_value": False},
    "lowercase": {"name": "Lowercase", "description": "Convertir a minúsculas", "requires_value": False},
    "cast_date": {"name": "Cast Date", "description": "Convertir a fecha", "requires_value": False},
    "groupby": {"name": "Group By + Agregación", "description": "Group by de columnas con agregaciones", "requires_value": True},
}

# Mock aggregation functions for when API is unavailable
MOCK_AGGREGATIONS = [
    {"id": "sum", "name": "Sum"},
    {"id": "mean", "name": "Mean"},
    {"id": "count", "name": "Count"},
    {"id": "min", "name": "Min"},
    {"id": "max", "name": "Max"},
    {"id": "first", "name": "First"},
    {"id": "last", "name": "Last"},
]

# Tipos de join disponibles para Gold layer
JOIN_TYPES = ["inner", "left"]


def get_mock_csv(table_name: str) -> Optional[str]:
    """Lee el CSV mock desde data/bronze/."""
    csv_path = BRONZE_PATH / f"{table_name}.csv"
    if csv_path.exists():
        return csv_path.read_text()
    return None


from typing import Union


def apply_mock_rule(df: pd.DataFrame, rule_id: str, column: str, value: Union[str, dict]) -> pd.DataFrame:
    """Apply a cleaning rule to the dataframe (mock/fallback)."""
    result = df.copy()
    
    if rule_id == "fillna":
        result[column] = result[column].fillna(value)
    elif rule_id == "trim":
        if result[column].dtype == "object":
            result[column] = result[column].str.strip()
    elif rule_id == "lowercase":
        if result[column].dtype == "object":
            result[column] = result[column].str.lower()
    elif rule_id == "cast_date":
        result[column] = pd.to_datetime(result[column], errors="coerce")
    elif rule_id == "groupby":
        # value is now a dict with group_columns and aggregations
        if isinstance(value, dict):
            result = groupby_agg(result, value.get("group_columns", []), value.get("aggregations", {}))
        else:
            # Fallback for old JSON format
            data = json.loads(value)
            result = groupby_agg(result, data["group_columns"], data["aggregations"])
    
    return result


def apply_mock_rules(df: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """Apply all rules in order to the dataframe (mock/fallback)."""
    result = df.copy()
    for rule in rules:
        rule_id = rule.get("operation") or rule.get("rule_id")
        params = rule.get("params", rule.get("parameters", {}))
        
        if rule_id == "groupby":
            # Special handling for groupby - pass full params
            result = apply_mock_rule(result, rule_id, "", params)
        else:
            # Standard handling - extract column and value
            column = params.get("column", "")
            value = params.get("value", "")
            result = apply_mock_rule(result, rule_id, column, value)
    return result


def apply_mock_join(left_df: pd.DataFrame, right_df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply a join between two dataframes (mock/fallback)."""
    left_key = config.get("left_key")
    right_key = config.get("right_key")
    join_type = config.get("join_type", "inner")
    
    if not left_key or not right_key:
        raise ValueError("Se requieren las columnas clave para el join")
    
    # Realizar el join
    result = pd.merge(
        left_df,
        right_df,
        left_on=left_key,
        right_on=right_key,
        how=join_type,
        suffixes=("_left", "_right")
    )
    
    return result
