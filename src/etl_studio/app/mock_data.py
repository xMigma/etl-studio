"""Mock data for development and testing when API is unavailable."""

from pathlib import Path

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


def get_mock_csv(table_name: str) -> str | None:
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
    elif rule_id == "groupby":
        data = json.loads(value)
        result = groupby_agg(result, data["group_columns"], data["aggregations"])
    
    return result


def apply_mock_rules(df: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """Apply all rules in order to the dataframe (mock/fallback)."""
    result = df.copy()
    for rule in rules:
        result = apply_mock_rule(result, rule["rule_id"], rule["column"], rule["value"])
    return result
