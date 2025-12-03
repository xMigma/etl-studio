"""Streamlit page for Bronze layer ingestion workflows."""

from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st
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
    
    # Fallback a CSV local
    mock_csv = get_mock_csv(table_name)
    if mock_csv:
        return pd.read_csv(io.StringIO(mock_csv)), True
    return None, True


def delete_table(table_name: str, is_mock: bool) -> tuple[bool, str]:
    """Delete a table via API. In mock mode, does nothing."""
    if is_mock:
        return False, "No se puede eliminar en modo mock"
    
    try:
        response = requests.delete(f"{API_BASE_URL}/bronze/tables/{table_name}", timeout=5)
        if response.status_code == 200:
            return True, f"Tabla '{table_name}' eliminada correctamente"
        return False, f"Error al eliminar: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Error de conexión: {e}"


@st.dialog("Detalle de Tabla", width="large")
def show_table_detail(table_name: str) -> None:
    """Display table details in a dialog."""
    with st.spinner(f"Cargando datos de {table_name}..."):
        df, is_mock = fetch_table_csv(table_name)
   
    if df is None:
        st.error(f"No se encontró información para la tabla '{table_name}'")
        return
    
    if is_mock:
        st.warning("Usando datos de prueba (API no disponible)")
    
    st.subheader(table_name)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filas", f"{len(df):,}")
    with col2:
        st.metric("Columnas", len(df.columns))
    
    st.write("**Tipos de datos:**")
    dtypes_df = df.dtypes.reset_index()
    dtypes_df.columns = ["Columna", "Tipo"]
    st.dataframe(dtypes_df, use_container_width=True, hide_index=True)
    
    st.write("**Datos:**")
    st.dataframe(df, use_container_width=True, hide_index=True, height=400)


def show() -> None:
    """Render the ingestion (Bronze) workspace."""

    st.header("Ingest · Bronze")
    st.write(
        "Placeholder for file uploads, schema validations, and ingestion monitoring."
    )
    uploaded_files = st.file_uploader("Sube tus tablas en formato CSV", type=["csv"], accept_multiple_files=True)
    
    if uploaded_files:
        files = [(file.name, file.getvalue()) for file in uploaded_files]
        try:
            response = requests.post(f"{API_BASE_URL}/bronze/upload", files=[("files", f) for f in files], timeout=30)
            if response.status_code == 200:
                st.success("Archivos subidos e ingeridos con éxito.")
            else:
                st.error("Error al subir los archivos.")
        except requests.exceptions.RequestException:
            st.warning("API no disponible. Los archivos no se pudieron subir.")
            
    
    st.subheader("Tablas Disponibles")
    
    tables, is_mock = fetch_tables()
    
    if is_mock:
        st.warning("Usando datos de prueba (API no disponible)")
    
    if not tables:
        st.info("No hay tablas disponibles en la capa Bronze.")
    else:
        with st.container(height=400):
            for table in tables:
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                with col1:
                    st.write(f"**{table['name']}**")
                with col2:
                    st.write(f"{table['rows']:,} filas")
                with col3:
                    if st.button("Ver", key=f"btn_ver_{table['name']}", use_container_width=True):
                        show_table_detail(table['name'])
                with col4:
                    if st.button("Eliminar", key=f"btn_del_{table['name']}", use_container_width=True, disabled=is_mock):
                        success, message = delete_table(table['name'], is_mock)
                        if success:
                            st.success(message)
                            st.rerun()
                        else:
                            st.error(message)
                st.divider()


if __name__ == "__main__":
    show()
