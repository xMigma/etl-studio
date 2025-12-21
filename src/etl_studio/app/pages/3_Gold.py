"""Streamlit page for Gold layer integration workflows."""

from __future__ import annotations
from typing import Optional
from io import StringIO

import pandas as pd
import streamlit as st

from etl_studio.app import setup_page
from etl_studio.app.components import show_table_detail_dialog
from etl_studio.app.data import fetch, fetch_table_csv, post
from etl_studio.app.mock_data import JOIN_TYPES, apply_mock_join

setup_page("Gold · ETL Studio")

def execute_join(
    left_table_name: str,
    right_table_name: str,
    config: dict[str, str],
    save: bool = False,
    output_table_name: str | None = None,
    left_source: str = "silver",
    right_source: str = "silver",
) -> pd.DataFrame | tuple[bool, str]:
    """Execute join operation. If save=True, saves to Gold and returns status."""
    payload = {
        "left_table": left_table_name,
        "right_table": right_table_name,
        "left_source": left_source,
        "right_source": right_source,
        "config": config,
    }
    
    endpoint = "apply" if save else "join"
    response, success = post("gold", endpoint, payload)
    
    if success and response:
        if save:
            table_name = output_table_name or f"{left_table_name}_{right_table_name}_joined"
            return True, f"Table '{table_name}' saved successfully"
        else:
            return pd.read_csv(StringIO(response))
    else:
        left_df = st.session_state.gold_dataframes.get(left_table_name)
        right_df = st.session_state.gold_dataframes.get(right_table_name)
        result_df = apply_mock_join(left_df, right_df, config)
        
        if save:
            table_name = output_table_name or f"{left_table_name}_{right_table_name}_joined"
            if "gold_tables" not in st.session_state:
                st.session_state.gold_tables = []
            st.session_state.gold_tables.append({"name": table_name, "rows": len(result_df)})
            if "gold_dataframes" not in st.session_state:
                st.session_state.gold_dataframes = {}
            st.session_state.gold_dataframes[table_name] = result_df
            return True, f"Table '{table_name}' saved successfully"
        else:
            return result_df


@st.dialog("Detalle de Tabla", width="large")
def show_table_detail(table_name: str, df: pd.DataFrame) -> None:
    """Display table details in a dialog."""
    show_table_detail_dialog(table_name, df=df)


def render_table_selector(
    label: str,
    options: list[str],
    table_dfs: dict[str, pd.DataFrame],
    key_prefix: str,
) -> tuple[Optional[str], Optional[str]]:
    """Render a table selector with key column selection and detail button.
    
    Returns:
        Tuple of (selected_table, selected_key_column)
    """
    st.markdown(f"**{label}**")
    table = st.selectbox("Tabla:", options, key=f"{key_prefix}_table_select")
    key_col = None
    
    if table and table in table_dfs:
        df = table_dfs[table]
        key_col = st.selectbox("Columna clave:", df.columns.tolist(), key=f"{key_prefix}_key_select")
        
        if st.button("Ver detalles", key=f"btn_ver_{key_prefix}", use_container_width=True, icon=":material/visibility:"):
            show_table_detail(table, df)
    
    return table, key_col


def show() -> None:
    """Render the integration (Gold) workspace."""

    st.header("Integration · Gold")
    
    # Inicializar estado
    if "gold_tables" not in st.session_state:
        st.session_state.gold_tables = []
    if "gold_dataframes" not in st.session_state:
        st.session_state.gold_dataframes = {}
    
    tables, is_mock = fetch("silver", "tables")
    
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
        df, _ = fetch_table_csv("silver", name)
        if df is not None:
            table_dfs[name] = df
    
    st.subheader("Configurar Join")
    
    col_left, col_join, col_right = st.columns([2, 1, 2])
    
    with col_left:
        left_table, left_key = render_table_selector("Tabla Izquierda", table_names, table_dfs, "left")
    
    with col_join:
        st.markdown("**Tipo de Join**")
        join_type = st.selectbox("Tipo:", JOIN_TYPES, key="join_type_select")
    
    with col_right:
        right_options = [t for t in table_names if t != left_table]
        right_table, right_key = render_table_selector("Tabla Derecha", right_options, table_dfs, "right")
    
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
    
    if all([left_table, right_table, left_key, right_key]):
        left_df = table_dfs.get(left_table)
        right_df = table_dfs.get(right_table)
        
        if left_df is not None and right_df is not None:
            try:
                config = {"left_key": left_key, "right_key": right_key, "join_type": join_type}
                result_df = execute_join(left_table, right_table, config, save=False)
                
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
                        success, message = execute_join(
                            left_table, right_table, config, save=True, output_table_name=table_name
                        )
                        if success:
                            st.success(message)
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
    
    gold_tables, _ = fetch("gold", "tables")
    gold_tables = gold_tables or st.session_state.get("gold_tables", [])
    
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
