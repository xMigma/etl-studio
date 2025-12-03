"""Mock data for development and testing when API is unavailable."""

from pathlib import Path

# Ruta a los CSVs de bronze para fallback
BRONZE_PATH = Path(__file__).parent.parent.parent.parent.parent / "data" / "bronze"

MOCK_TABLES = [
    {"name": "customers", "rows": 5},
    {"name": "orders", "rows": 7},
    {"name": "products", "rows": 5},
]


def get_mock_csv(table_name: str) -> str | None:
    """Lee el CSV mock desde data/bronze/."""
    csv_path = BRONZE_PATH / f"{table_name}.csv"
    if csv_path.exists():
        return csv_path.read_text()
    return None
