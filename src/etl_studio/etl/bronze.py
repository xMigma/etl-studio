"""Bronze layer ingestion helpers."""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from etl_studio.app.pages.mock_data import MOCK_TABLES, get_mock_csv


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def fetch_tables() -> tuple[list, bool]:
    """Fetch tables from API, fallback to mock data on failure."""
    try:
        response = requests.get(f"{API_BASE_URL}/bronze/tables", timeout=5)
        if response.status_code == 200:
            return response.json(), False
    except requests.exceptions.RequestException:
        pass
    return MOCK_TABLES, True


def fetch_table_csv(table_name: str) -> tuple[pd.DataFrame | None, bool]:
    """Fetch table CSV from API, fallback to mock CSV on failure."""
    try:
        response = requests.get(f"{API_BASE_URL}/bronze/tables/{table_name}", timeout=5)
        if response.status_code == 200:
            return pd.read_csv(io.StringIO(response.text)), False
    except requests.exceptions.RequestException:
        pass
    
    mock_csv = get_mock_csv(table_name)
    if mock_csv:
        return pd.read_csv(io.StringIO(mock_csv)), True
    return None, True


def load_csv_to_bronze(file_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a raw CSV file into the bronze zone and capture metadata."""

    # TODO: Implement real ingestion logic (validation, persistence, metadata capture).
    df = pd.DataFrame()
    metadata = {"rows": len(df), "columns": list(df.columns), "source": str(file_path)}
    print(f"[Bronze] Ingest placeholder for {file_path}")
    return df, metadata
