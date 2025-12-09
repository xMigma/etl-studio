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

    if not gold_tables:
        st.warning("No hay tablas disponibles en Gold. Crea primero datasets en Gold Layer.")
        if st.button("Ir a Gold", icon=":material/arrow_forward:"):
            st.switch_page("pages/3_Gold.py")
        return
    
    table_names = [t["name"] for t in gold_tables]
    selected_table = st.selectbox("Selecciona el dataset:", table_names, key="dataset_select")
    
    if not selected_table:
        return
    
        # Cargar datos
    with st.spinner("Cargando dataset..."):
        df, _ = fetch_table_csv("gold", selected_table)
    
    if df is None:
        st.error("No se pudo cargar el dataset")
        return
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filas", f"{len(df):,}")
    with col2:
        st.metric("Columnas", len(df.columns))
    
    with st.expander("Vista previa del dataset", expanded=False):
        st.dataframe(df.head(20), use_container_width=True, height=300)
    
    st.divider()

if __name__ == "__main__":
    show()