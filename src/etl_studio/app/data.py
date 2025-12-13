"""Centralized data fetching functions for all ETL layers."""

from __future__ import annotations

import io
from typing import Any, Literal, Optional

import pandas as pd
import requests

from etl_studio.app.mock_data import MOCK_TABLES, MOCK_RULES, get_mock_csv
from etl_studio.config import API_BASE_URL


Layer = Literal["bronze", "silver", "gold"]
Resource = Literal["tables", "rules"]

# Mock data por recurso
MOCK_DATA: dict[str, Any] = {
    "tables": MOCK_TABLES,
    "rules": MOCK_RULES,
}


def fetch(layer: Layer, resource: Resource) -> tuple[Any, bool]:
    """Fetch a resource from a layer. Falls back to mock data if API unavailable."""
    try:
        response = requests.get(f"{API_BASE_URL}/{layer}/{resource}", timeout=5)
        if response.status_code == 200:
            return response.json(), False
    except requests.exceptions.RequestException:
        pass
    return MOCK_DATA.get(resource), True


def post(
    layer: Layer, 
    resource: str, 
    payload: Optional[dict] = None, 
    files: Optional[list[tuple[str, bytes]]] = None,
    timeout: int = 30
) -> tuple[Any, bool]:
    """POST to a layer endpoint. Can send JSON payload or files. Returns (response_json, success)."""
    try:
        kwargs: dict[str, Any] = {"timeout": timeout}
        if files:
            kwargs["files"] = [("files", file_data) for file_data in files]
        elif payload:
            kwargs["json"] = payload
        
        response = requests.post(f"{API_BASE_URL}/{layer}/{resource}", **kwargs)
        if response.status_code == 201:
            return response.json(), True
    except requests.exceptions.RequestException:
        pass
    return None, False


def fetch_table_csv(layer: Layer, table_name: str, preview: bool = False) -> tuple[Optional[pd.DataFrame], bool]:
    """Fetch table CSV from a layer. Falls back to mock CSV if API unavailable."""
    try:
        response = requests.get(f"{API_BASE_URL}/{layer}/tables/{table_name}?preview={str(preview).lower()}", timeout=5)
        if response.status_code == 200:
            return pd.read_csv(io.StringIO(response.text)), False
    except requests.exceptions.RequestException:
        pass
    
    mock_csv = get_mock_csv(table_name)
    if mock_csv:
        return pd.read_csv(io.StringIO(mock_csv)), True
    return None, True


def delete(layer: Layer, resource: str, name: str) -> tuple[bool, str]:
    """Delete a resource from a layer."""
    try:
        response = requests.delete(f"{API_BASE_URL}/{layer}/{resource}/{name}", timeout=5)
        if response.status_code == 200:
            return True, f"'{name}' eliminado"
        return False, f"Error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Error de conexión: {e}"

