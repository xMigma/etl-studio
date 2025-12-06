"""Streamlit page for Bronze layer ingestion workflows."""

from __future__ import annotations

import requests
import streamlit as st

import pandas as pd

from etl_studio.app import setup_page
from etl_studio.app.components import render_table_detail
from etl_studio.config import API_BASE_URL
from etl_studio.etl.bronze import fetch_tables, fetch_table_csv

setup_page("Bronze · ETL Studio")


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
   
    # Contenedor principal con márgenes laterales
    _, main_col, _ = st.columns([0.5, 9, 0.5])
    
    with main_col:
        render_table_detail(df, table_name, is_mock)
    


def show() -> None:
    """Render the ingestion (Bronze) workspace."""

    st.header("Ingest · Bronze")
    
    st.subheader("Subir archivos")
    uploaded_files = st.file_uploader(
        "Arrastra o selecciona archivos CSV",
        type=["csv"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        files = [(file.name, file.getvalue()) for file in uploaded_files]
        try:
            response = requests.post(f"{API_BASE_URL}/bronze/upload", files=[("files", f) for f in files], timeout=30)
            if response.status_code == 200:
                st.success("Archivos subidos e ingeridos con éxito.")
                st.rerun()
            else:
                st.error("Error al subir los archivos.")
        except requests.exceptions.RequestException:
            st.warning("API no disponible. Los archivos no se pudieron subir.")
    
    st.divider()
    
    st.subheader("Tablas disponibles")
    
    tables, is_mock = fetch_tables()
    
    if is_mock:
        st.info("Modo de prueba: API no disponible")
    
    if not tables:
        st.info("No hay tablas disponibles.")
    else:
        with st.container(height=350):
            for table in tables:
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                    with col1:
                        st.write(f"**{table['name']}**")
                    with col2:
                        st.caption(f"{table['rows']:,} filas")
                    with col3:
                        if st.button("Ver", key=f"btn_ver_{table['name']}", use_container_width=True):
                            show_table_detail(table['name'])
                    with col4:
                        if st.button("Eliminar", key=f"btn_del_{table['name']}", use_container_width=True, disabled=is_mock, type="secondary"):
                            success, message = delete_table(table['name'], is_mock)
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
    
    st.divider()
    
    if st.button("Continuar a Silver", type="primary"):
        st.switch_page("pages/2_Silver.py")


if __name__ == "__main__":
    show()
