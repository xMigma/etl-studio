"""Mock data for development and testing when API is unavailable."""

from pathlib import Path
from typing import Any, Optional

import pandas as pd

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
}

# Tipos de join disponibles para Gold layer
JOIN_TYPES = ["inner", "left"]


def get_mock_csv(table_name: str) -> Optional[str]:
    """Lee el CSV mock desde data/bronze/."""
    csv_path = BRONZE_PATH / f"{table_name}.csv"
    if csv_path.exists():
        return csv_path.read_text()
    return None


def apply_mock_rule(df: pd.DataFrame, rule_id: str, column: str, value: str) -> pd.DataFrame:
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
    
    return result


def apply_mock_rules(df: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """Apply all rules in order to the dataframe (mock/fallback)."""
    result = df.copy()
    for rule in rules:
        result = apply_mock_rule(result, rule["rule_id"], rule["column"], rule["value"])
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
