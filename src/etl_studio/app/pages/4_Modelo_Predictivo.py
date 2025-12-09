"""Streamlit page for supervised ML workflows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from etl_studio.app import setup_page
from etl_studio.app.data import fetch, fetch_table_csv
from etl_studio.ai import model

setup_page("Modelo Predictivo · ETL Studio")

def show() -> None:
    """Render the model training and inference workspace."""
    
    st.header("Machine Learning Model Training")
    
    # Seleccionar dataset de Gold
    st.subheader("Dataset Selection")
    
    gold_tables, is_mock = fetch("gold", "tables")


if __name__ == "__main__":
    show()