"""Streamlit page for Gold layer integration workflows."""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from etl_studio.app import setup_page
from etl_studio.app.components import render_table_detail
from etl_studio.app.mock_data import JOIN_TYPES, apply_mock_join
from etl_studio.config import API_BASE_URL
from etl_studio.etl.bronze import fetch_tables, fetch_table_csv

setup_page("Gold · ETL Studio")


def fetch_silver_tables() -> tuple[list[dict], bool]:
    """Fetch tables from Silver layer, fallback to Bronze tables for mock."""
    try:
        response = requests.get(f"{API_BASE_URL}/silver/tables", timeout=5)
        if response.status_code == 200:
            return response.json(), False
    except requests.exceptions.RequestException:
        pass
    # Fallback: usar tablas de bronze como mock
    return fetch_tables()


def fetch_gold_tables() -> list[dict] | None:
    """Fetch existing Gold tables."""
    try:
        response = requests.get(f"{API_BASE_URL}/gold/tables", timeout=5)
        if response.status_code == 200:
            return response.json()
    except requests.exceptions.RequestException:
        pass
    return None


def save_gold_table(df: pd.DataFrame, name: str) -> bool:
    """Save a Gold table via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/gold/tables",
            json={"name": name, "data": df.to_dict(orient="records")},
            timeout=10
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        # Mock: guardar en session state
        if "gold_tables" not in st.session_state:
            st.session_state.gold_tables = []
        st.session_state.gold_tables.append({"name": name, "rows": len(df)})
        if "gold_dataframes" not in st.session_state:
            st.session_state.gold_dataframes = {}
        st.session_state.gold_dataframes[name] = df
        return True


@st.dialog("Detalle de Tabla", width="large")
def show_table_detail(table_name: str, df: pd.DataFrame) -> None:
    """Display table details in a dialog."""
    _, main_col, _ = st.columns([0.5, 9, 0.5])
    with main_col:
        render_table_detail(df, table_name, is_mock=True)


def show() -> None:
    """Render the integration (Gold) workspace."""

    st.header("Integration · Gold")
    
    # Inicializar estado
    if "gold_tables" not in st.session_state:
        st.session_state.gold_tables = []
    if "gold_dataframes" not in st.session_state:
        st.session_state.gold_dataframes = {}
    
    tables, is_mock = fetch_silver_tables()
    
    if is_mock:
        st.info("Modo de prueba: API no disponible")
    
    if not tables:
        st.warning("No hay tablas disponibles en Silver. Procesa primero las tablas en la capa Silver.")
        if st.button("Ir a Silver", type="primary"):
            st.switch_page("pages/2_Silver.py")
        return
    
    table_names = [t["name"] for t in tables]
    
    # Cargar DataFrames de las tablas
    table_dfs = {}
    for name in table_names:
        df, _ = fetch_table_csv(name)
        if df is not None:
            table_dfs[name] = df
    
    st.subheader("Configurar Join")
    
    col_left, col_join, col_right = st.columns([2, 1, 2])
    
    with col_left:
        st.markdown("**Tabla Izquierda**")
        left_table = st.selectbox("Tabla:", table_names, key="left_table_select")
        
        if left_table and left_table in table_dfs:
            left_df = table_dfs[left_table]
            left_key = st.selectbox("Columna clave:", left_df.columns.tolist(), key="left_key_select")
            
            with st.expander("Vista previa", expanded=False):
                st.dataframe(left_df.head(5), use_container_width=True)
    
    with col_join:
        st.markdown("**Tipo de Join**")
        join_type = st.selectbox("Tipo:", JOIN_TYPES, key="join_type_select")
    
    with col_right:
        st.markdown("**Tabla Derecha**")
        right_options = [t for t in table_names if t != left_table]
        right_table = st.selectbox("Tabla:", right_options, key="right_table_select")
        
        if right_table and right_table in table_dfs:
            right_df = table_dfs[right_table]
            right_key = st.selectbox("Columna clave:", right_df.columns.tolist(), key="right_key_select")
            
            with st.expander("Vista previa", expanded=False):
                st.dataframe(right_df.head(5), use_container_width=True)
    
    st.divider()
    
    # Nombre de la tabla de salida
    col_name, _ = st.columns([2, 3])
    with col_name:
        output_name = st.text_input(
            "Nombre de la tabla resultante:",
            value=f"{left_table}_{right_table}" if left_table and right_table else "joined_table",
            key="output_name_input"
        )
    
    st.divider()
    
    # Preview del resultado
    st.subheader("Preview del Join")
    
    # Obtener valores directamente de session_state
    left_table = st.session_state.get("left_table_select")
    right_table = st.session_state.get("right_table_select")
    left_key = st.session_state.get("left_key_select")
    right_key = st.session_state.get("right_key_select")
    join_type = st.session_state.get("join_type_select", "inner")
    output_name = st.session_state.get("output_name_input", "joined_table")
    
    if all([left_table, right_table, left_key, right_key]):
        left_df = table_dfs.get(left_table)
        right_df = table_dfs.get(right_table)
        
        if left_df is not None and right_df is not None:
            try:
                config = {"left_key": left_key, "right_key": right_key, "join_type": join_type}
                result_df = apply_mock_join(left_df, right_df, config)
                
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.metric("Filas tabla izquierda", len(left_df))
                with col_info2:
                    st.metric("Filas tabla derecha", len(right_df))
                with col_info3:
                    st.metric("Filas resultado", len(result_df))
                
                st.dataframe(result_df.head(20), use_container_width=True, height=350)
                
                st.divider()
                
                col_save, col_continue = st.columns([1, 1])
                
                with col_save:
                    if st.button("Guardar tabla Gold", type="primary", use_container_width=True, icon=":material/save:"):
                        table_name = output_name or f"{left_table}_{right_table}"
                        success = save_gold_table(result_df, table_name)
                        if success:
                            st.success(f"Tabla '{table_name}' guardada en la capa Gold")
                        else:
                            st.error("Error al guardar la tabla")
                
                with col_continue:
                    if st.button("Continuar a Modelo Predictivo", use_container_width=True, icon=":material/arrow_forward:"):
                        st.switch_page("pages/4_Modelo_Predictivo.py")
                        
            except Exception as e:
                st.error(f"Error al realizar el join: {e}")
        else:
            st.warning("No se pudieron cargar las tablas seleccionadas")
    else:
        st.info("Configura las tablas y columnas clave para ver el preview del join")
    
    st.divider()
    
    # Mostrar tablas Gold existentes
    st.subheader("Tablas Gold existentes")
    
    gold_tables = fetch_gold_tables() or st.session_state.get("gold_tables", [])
    
    if not gold_tables:
        st.caption("No hay tablas Gold creadas aún")
    else:
        with st.container(height=200):
            for table in gold_tables:
                with st.container(border=True):
                    col1, col2, col3 = st.columns([3, 2, 1])
                    with col1:
                        st.write(f"**{table['name']}**")
                    with col2:
                        st.caption(f"{table['rows']:,} filas")
                    with col3:
                        gold_df = st.session_state.gold_dataframes.get(table["name"])
                        if gold_df is not None:
                            if st.button("Ver", key=f"btn_ver_gold_{table['name']}", use_container_width=True):
                                show_table_detail(table["name"], gold_df)


if __name__ == "__main__":
    show()
