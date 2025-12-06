"""Reusable UI components for the ETL Studio platform."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import streamlit as st

if TYPE_CHECKING:
    from etl_studio.app.data import Layer


def render_table_metrics(df: pd.DataFrame) -> None:
    """Render row and column count metrics."""
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Filas", f"{len(df):,}")
    with col2:
        st.metric("Columnas", len(df.columns))


def render_data_types(df: pd.DataFrame) -> None:
    """Render data types expander."""
    with st.expander("Tipos de datos"):
        dtypes_df = df.dtypes.reset_index()
        dtypes_df.columns = ["Columna", "Tipo"]
        st.dataframe(dtypes_df, use_container_width=True, hide_index=True)


def render_data_preview(df: pd.DataFrame, expanded: bool = True, height: int = 400) -> None:
    """Render data preview expander."""
    with st.expander("Datos", expanded=expanded):
        st.dataframe(df, use_container_width=True, hide_index=True, height=height)


def render_null_values(df: pd.DataFrame) -> None:
    """Render null values expander."""
    with st.expander("Valores nulos"):
        nulls = df.isnull().sum().reset_index()
        nulls.columns = ["Columna", "Nulos"]
        st.dataframe(nulls, use_container_width=True, hide_index=True)


def render_basic_stats(df: pd.DataFrame) -> None:
    """Render basic statistics expander."""
    with st.expander("Estadísticas básicas"):
        st.dataframe(df.describe(), use_container_width=True)


def render_distributions(df: pd.DataFrame, key_prefix: str = "dist") -> None:
    """Render distribution charts expander."""
    with st.expander("Distribuciones"):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if numeric_cols:
            st.write("**Variables numéricas (histogramas)**")
            selected_num = st.selectbox(
                "Selecciona columna numérica", 
                numeric_cols, 
                key=f"hist_{key_prefix}"
            )
            if selected_num:
                st.bar_chart(df[selected_num].dropna().value_counts().sort_index())
        
        if categorical_cols:
            st.write("**Variables categóricas (bar chart)**")
            selected_cat = st.selectbox(
                "Selecciona columna categórica", 
                categorical_cols, 
                key=f"bar_{key_prefix}"
            )
            if selected_cat:
                value_counts = df[selected_cat].value_counts().head(20)
                st.bar_chart(value_counts)
        
        if not numeric_cols and not categorical_cols:
            st.info("No hay columnas disponibles para visualizar")


def render_duplicates_check(df: pd.DataFrame) -> None:
    """Render duplicate rows detection."""
    st.write("**Filas duplicadas**")
    duplicates = df.duplicated().sum()
    total_rows = len(df)
    if duplicates > 0:
        st.warning(f":material/warning: {duplicates:,} filas duplicadas ({duplicates/total_rows*100:.1f}%)")
    else:
        st.success(":material/check_circle: No hay filas duplicadas")


def render_constant_cols_check(df: pd.DataFrame) -> None:
    """Render constant columns detection."""
    st.write("**Columnas constantes**")
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    if constant_cols:
        st.warning(f":material/warning: {len(constant_cols)} columna(s) constante(s):")
        for col in constant_cols:
            st.caption(f"• {col}")
    else:
        st.success(":material/check_circle: No hay columnas constantes")


def render_high_cardinality_check(df: pd.DataFrame) -> None:
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


def render_quick_detections(df: pd.DataFrame) -> None:
    """Render quick detections expander."""
    with st.expander("Detecciones rápidas"):
        col1, col2 = st.columns(2)
        
        with col1:
            render_duplicates_check(df)
        
        with col2:
            render_constant_cols_check(df)
        
        st.divider()
        render_high_cardinality_check(df)


def render_table_detail(df: pd.DataFrame | None, table_name: str, is_mock: bool = False) -> None:
    """Render full table detail view with all analysis components.
    
    Args:
        df: DataFrame to analyze (can be None)
        table_name: Name of the table (used for display and widget keys)
        is_mock: Whether the data is from mock/test source
    """
    if df is None:
        st.error(f"No se encontró información para la tabla '{table_name}'")
        return
    
    if is_mock:
        st.warning("Usando datos de prueba (API no disponible)")
    
    st.subheader(table_name)
    render_table_metrics(df)
    render_data_types(df)
    render_data_preview(df)
    render_null_values(df)
    render_basic_stats(df)
    render_distributions(df, key_prefix=table_name)
    render_quick_detections(df)


def show_table_detail_dialog(
    table_name: str,
    layer: "Layer | None" = None,
    df: pd.DataFrame | None = None,
) -> None:
    """Display table details in a modal dialog.
    
    Args:
        table_name: Name of the table to display
        layer: Layer to fetch from (required if df is None)
        df: Pre-loaded DataFrame (if None, fetches from layer)
    """
    from etl_studio.app.data import fetch_table_csv
    
    is_mock = False
    if df is None:
        if layer is None:
            st.error("Debe especificarse 'layer' o 'df'")
            return
        with st.spinner(f"Cargando datos de {table_name}..."):
            df, is_mock = fetch_table_csv(layer, table_name)
    
    _, main_col, _ = st.columns([0.5, 9, 0.5])
    with main_col:
        render_table_detail(df, table_name, is_mock)
