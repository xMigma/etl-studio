"""Streamlit page for Silver layer data cleaning workflows."""

from __future__ import annotations
import os

import streamlit as st
import requests

from etl_studio.app.pages.mock_data import MOCK_TABLES


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


def show() -> None:
    """Render the cleaning (Silver) workspace."""

    st.header("Cleaning · Silver")
    st.write(
        "Placeholder for transformation recipes, profiling charts, and quality gates."
    )
    
    tables, is_mock = fetch_tables()
    
    if is_mock:
        st.info("Modo de prueba: API no disponible")
    
    table_names = [table['name'] for table in tables]
    
    options = st.selectbox(
        "Select the table you want to clean:",
        table_names,
    )
    
    


if __name__ == "__main__":
    show()
