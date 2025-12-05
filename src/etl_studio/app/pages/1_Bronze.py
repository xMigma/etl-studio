"""Streamlit page for Bronze layer ingestion workflows."""

from __future__ import annotations

import requests
import streamlit as st

import pandas as pd

from etl_studio.app import setup_page
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


def _render_table_metrics(df: pd.DataFrame) -> None:
    """Render row and column count metrics."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filas", f"{len(df):,}")
    with col2:
        st.metric("Columnas", len(df.columns))


def _render_data_types(df: pd.DataFrame) -> None:
    """Render data types expander."""
    with st.expander("Tipos de datos"):
        dtypes_df = df.dtypes.reset_index()
        dtypes_df.columns = ["Columna", "Tipo"]
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)


def _render_data_preview(df: pd.DataFrame) -> None:
    """Render data preview expander."""
    with st.expander("Datos", expanded=True):
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)


def _render_null_values(df: pd.DataFrame) -> None:
    """Render null values expander."""
    with st.expander("Valores nulos"):
        nulls = df.isnull().sum().reset_index()
        nulls.columns = ["Columna", "Nulos"]
        st.dataframe(nulls, use_container_width=True, hide_index=True)


def _render_basic_stats(df: pd.DataFrame) -> None:
    """Render basic statistics expander."""
    with st.expander("Estadísticas básicas"):
        st.dataframe(df.describe(), use_container_width=True)


def _render_distributions(df: pd.DataFrame, table_name: str) -> None:
    """Render distribution charts expander."""
    with st.expander("Distribuciones"):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols:
            st.write("**Variables numéricas (histogramas)**")
            selected_num = st.selectbox("Selecciona columna numérica", numeric_cols, key=f"hist_{table_name}")
            if selected_num:
                st.bar_chart(df[selected_num].dropna().value_counts().sort_index())
        
        if categorical_cols:
            st.write("**Variables categóricas (bar chart)**")
            selected_cat = st.selectbox("Selecciona columna categórica", categorical_cols, key=f"bar_{table_name}")
            if selected_cat:
                value_counts = df[selected_cat].value_counts().head(20)
                st.bar_chart(value_counts)
        
        if not numeric_cols and not categorical_cols:
            st.info("No hay columnas disponibles para visualizar")


def _render_duplicates_check(df: pd.DataFrame) -> None:
    """Render duplicate rows detection."""
    st.write("**Filas duplicadas**")
    duplicates = df.duplicated().sum()
    total_rows = len(df)
    if duplicates > 0:
        st.warning(f":material/warning: {duplicates:,} filas duplicadas ({duplicates/total_rows*100:.1f}%)")
    else:
        st.success(":material/check_circle: No hay filas duplicadas")


def _render_constant_cols_check(df: pd.DataFrame) -> None:
    """Render constant columns detection."""
    st.write("**Columnas constantes**")
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        st.warning(f":material/warning: {len(constant_cols)} columna(s) constante(s):")
        for col in constant_cols:
            st.caption(f"• {col}")
    else:
        st.success(":material/check_circle: No hay columnas constantes")


def _render_high_cardinality_check(df: pd.DataFrame) -> None:
    """Render high cardinality columns detection."""
    st.write("**Columnas con alta cardinalidad**")
    high_cardinality = []
    for col in df.columns:
        unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
        if unique_ratio > 0.9 and df[col].nunique() > 10:
            high_cardinality.append((col, df[col].nunique(), unique_ratio))
    
    if high_cardinality:
        st.info(f":material/info: {len(high_cardinality)} columna(s) con alta cardinalidad (>90% valores únicos):")
        for col, unique, ratio in high_cardinality:
            st.caption(f"• {col}: {unique:,} valores únicos ({ratio*100:.1f}%)")
    else:
        st.success(":material/check_circle: No hay columnas con alta cardinalidad")


def _render_quick_detections(df: pd.DataFrame) -> None:
    """Render quick detections expander."""
    with st.expander("Detecciones rápidas"):
        col1, col2 = st.columns(2)
        
        with col1:
            _render_duplicates_check(df)
        
        with col2:
            _render_constant_cols_check(df)
        
        st.divider()
        _render_high_cardinality_check(df)


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
    
    # Contenedor principal con márgenes laterales
    _, main_col, _ = st.columns([0.5, 9, 0.5])
    
    with main_col:
        st.subheader(table_name)
        _render_table_metrics(df)
        _render_data_types(df)
        _render_data_preview(df)
        _render_null_values(df)
        _render_basic_stats(df)
        _render_distributions(df, table_name)
        _render_quick_detections(df)
    


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
