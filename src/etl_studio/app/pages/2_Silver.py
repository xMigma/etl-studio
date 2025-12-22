"""Streamlit page for Silver layer data cleaning workflows."""

from __future__ import annotations

import pandas as pd
import streamlit as st
import json

from etl_studio.app import setup_page
from etl_studio.app.components import show_table_detail_dialog, show_confirm_delete_dialog
from etl_studio.app.data import fetch, fetch_table_csv, post, fetch_aggregations
from etl_studio.app.mock_data import apply_mock_rules

setup_page("Silver · ETL Studio")


def rules_to_operations(rules: list[dict]) -> list[dict]:
    """Convert rules to API operations format."""
    operations = []
    for rule in rules:
        if rule["rule_id"] == "groupby":
            # Special handling for groupby - expects group_columns and aggregations
            operations.append({
                "operation": "groupby",
                "params": {
                    "group_columns": rule["parameters"].get("group_columns", []),
                    "aggregations": rule["parameters"].get("aggregations", {})
                }
            })
        else:
            # Standard handling for other rules
            operations.append({
                "operation": rule["rule_id"],
                "params": {
                    "column": rule["parameters"].get("column", ""),
                    "value": rule["parameters"].get("value", ""),
                    "new_name": rule["parameters"].get("new_name", "")
                }
            })
    return operations


def fetch_preview(table: str, rules: list[dict], df: pd.DataFrame) -> pd.DataFrame:
    """Fetch preview from API, fallback to local processing on failure."""
    operations = rules_to_operations(rules)
    payload = {"table_name": table, "operations": operations}
    data, success = post("silver", "preview", payload)
    
    if success:
        from io import StringIO
        df_preview = pd.read_csv(StringIO(data))
    else:
        df_preview = apply_mock_rules(df, operations)    
        
    return df_preview


def get_applied_rules(table_name: str) -> list[dict]:
    """Get applied rules for a table from session state."""
    if "applied_rules" not in st.session_state:
        st.session_state.applied_rules = {}
    return st.session_state.applied_rules.get(table_name, [])


def update_active_dataframe(table_name: str, df_original: pd.DataFrame) -> None:
    """Update the active dataframe based on applied rules."""
    applied_rules = get_applied_rules(table_name)
    
    if applied_rules:
        st.session_state.active_dataframe = fetch_preview(table_name, applied_rules, df_original)
    else:
        st.session_state.active_dataframe = df_original.copy()


def add_rule_to_table(table_name: str, rule_id: str, parameters: dict, df_original: pd.DataFrame) -> None:
    """Add a rule to the table's applied rules."""
    if "applied_rules" not in st.session_state:
        st.session_state.applied_rules = {}
    
    if table_name not in st.session_state.applied_rules:
        st.session_state.applied_rules[table_name] = []
        
    table = st.session_state.applied_rules[table_name]
    exists = any(r["rule_id"] == rule_id and r["parameters"] == parameters for r in table) 
    
    if not exists:
        st.session_state.applied_rules[table_name].append({
            "rule_id": rule_id,
            "parameters": parameters,
        })       
        
    update_active_dataframe(table_name, df_original)

def remove_rule_from_table(table_name: str, index: int, df_original: pd.DataFrame) -> None:
    """Remove a rule from the table's applied rules by index."""
    if "applied_rules" in st.session_state and table_name in st.session_state.applied_rules:
        if 0 <= index < len(st.session_state.applied_rules[table_name]):
            st.session_state.applied_rules[table_name].pop(index)
            update_active_dataframe(table_name, df_original)


def clear_rules_for_table(table_name: str, df_original: pd.DataFrame) -> None:
    """Clear all rules for a table."""
    if "applied_rules" in st.session_state and table_name in st.session_state.applied_rules:
        st.session_state.applied_rules[table_name] = []
        update_active_dataframe(table_name, df_original)


@st.dialog("Detalle de Tabla", width="large")
def show_table_detail(table_name: str, layer: str = "bronze") -> None:
    """Display table details in a dialog."""
    show_table_detail_dialog(table_name, layer=layer)


@st.dialog("Confirmar eliminación")
def confirm_delete(table_name: str, layer: str = "silver") -> None:
    """Display confirmation dialog for table deletion."""
    show_confirm_delete_dialog(table_name, layer=layer)


def show() -> None:
    """Render the cleaning (Silver) workspace."""
    setup_page("Cleaning · Silver")

    st.header("Cleaning · Silver")
    
    # Inicializar estado
    if "selected_rule" not in st.session_state:
        st.session_state.selected_rule = None
    if "applied_rules" not in st.session_state:
        st.session_state.applied_rules = {}
    
    tables, tables_mock = fetch("bronze", "tables")
    available_rules, rules_mock = fetch("silver", "rules")
    
    if tables_mock or rules_mock:
        st.info("Modo de prueba: API no disponible")
    
    table_names = [table['name'] for table in tables]
    
    col_select, col_detail = st.columns([4, 1], vertical_alignment="bottom")
    
    with col_select:
        selected_table = st.selectbox(
            "Selecciona la tabla a limpiar:",
            table_names,
        )
    
    with col_detail:
        if selected_table and st.button("Ver detalles", use_container_width=True, icon=":material/visibility:"):
            show_table_detail(selected_table)
    
    if not selected_table:
        return
    
    # Cargar datos de la tabla
    df, _ = fetch_table_csv("bronze", selected_table)
    if df is None:
        st.error("No se pudo cargar la tabla")
        return
    
    if "active_dataframe" not in st.session_state or "last_table" not in st.session_state or st.session_state.last_table != selected_table:
        st.session_state.last_table = selected_table
        update_active_dataframe(selected_table, df)
    
    st.divider()
    
    # Layout de 2 columnas para reglas y configuración
    col_rules, col_editor, col_applied = st.columns([1.5, 1.5, 1.5], gap="large")
    
    with col_rules:
        st.subheader("Reglas")
        
        for rule_id, rule_data in available_rules["rules"].items():
            is_selected = st.session_state.selected_rule == rule_id
            button_type = "primary" if is_selected else "secondary"
            
            if st.button(
                rule_data.get('name', rule_id),
                key=f"btn_{rule_id}",
                use_container_width=True,
                type=button_type
            ):
                st.session_state.selected_rule = rule_id
                st.rerun()
    
    with col_editor:
        st.subheader("Configuración")
        
        if st.session_state.selected_rule:
            rule_id = st.session_state.selected_rule
            rule = available_rules["rules"].get(rule_id)
            
            if rule:
                if rule_id == "groupby":
                    st.caption("Configura el Group By con agregaciones")
                    
                    aggregations_list, _ = fetch_aggregations()
                    agg_options = [agg["id"] for agg in aggregations_list]
                    
                    if "groupby_aggregations" not in st.session_state:
                        st.session_state.groupby_aggregations = {}
                    
                    # paso 1, muestra las columnas de agrupacion
                    group_columns = st.multiselect(
                        "Columnas para agrupar:",
                        st.session_state.active_dataframe.columns.tolist(),
                        key="groupby_columns"
                    )
                    
                    # paso 2, muestra las columnas de agregacion
                    if group_columns:
                        st.divider()
                        st.caption("Añadir agregaciones:")
                        
                        remaining_cols = [c for c in st.session_state.active_dataframe.columns if c not in group_columns]
                        
                        col1, col2, col3 = st.columns([2, 2, 1], vertical_alignment="bottom")
                        with col1:
                            selected_cols = st.selectbox(
                                "Columna:",
                                remaining_cols,
                                key="agg_cols_select"
                            )
                        with col2:
                            selected_agg = st.selectbox(
                                "Función:",
                                agg_options,
                                key="agg_func_select"
                            )
                        with col3:
                            if st.button("", help="Añadir agregación", use_container_width=True, icon=":material/add_circle:"):
                                if selected_cols: 
                                    st.session_state.groupby_aggregations[selected_cols] = selected_agg
                                    st.rerun()
                        
                        if st.session_state.groupby_aggregations:
                            st.divider()
                            st.caption("Agregaciones configuradas:")
                            for col, func in list(st.session_state.groupby_aggregations.items()):
                                col_agg, col_del = st.columns([4, 1])
                                with col_agg:
                                    st.text(f"• {col}: {func}")
                                with col_del:
                                    if st.button("🗑️", key=f"del_agg_{col}", help="Eliminar"):
                                        del st.session_state.groupby_aggregations[col]
                                        st.rerun()
                        
                        parameters = {
                            "group_columns": group_columns,
                            "aggregations": st.session_state.groupby_aggregations
                        }
                        
                        if st.button("Añadir Regla", type="primary", use_container_width=True, icon=":material/add:", disabled=not st.session_state.groupby_aggregations):
                            add_rule_to_table(selected_table, rule_id, parameters, df)
                            st.session_state.groupby_aggregations = {}
                            st.rerun()
                    else:
                        st.info("Selecciona al menos una columna para agrupar")
                
                else:
                    # Standard handling for other rules
                    column = st.selectbox("Columna:", st.session_state.active_dataframe.columns.tolist(), key="rule_column")     
                    parameters = {}
                    
                    parameters["column"] = column
                    
                    # Los parámetros vienen como lista de strings: ["column", "new_name"]
                    # El primero (column) ya se maneja con el selectbox, los demás necesitan inputs
                    rule_params = rule.get("parameters", [])
                    for param_name in rule_params[1:]:  # Saltar el primer parámetro (column)
                        # Generar un label legible: "new_name" -> "New Name"
                        label = param_name.replace("_", " ").title()
                        parameters[param_name] = st.text_input(label, key=f"param_{param_name}")
                    
                    if st.button("Añadir", type="primary", use_container_width=True, icon=":material/add:"):
                        add_rule_to_table(selected_table, rule_id, parameters, df)
                        st.rerun()
        else:
            st.caption("Selecciona una regla para configurarla")
    
    with col_applied:
        st.subheader("Reglas aplicadas")
        
        applied_rules = get_applied_rules(selected_table)
        if applied_rules:
            for i, r in enumerate(applied_rules):
                rule_data = available_rules["rules"].get(r["rule_id"])
                rule_name = rule_data["name"] if rule_data else r["rule_id"]
                col_rule, col_delete = st.columns([4, 1])
                with col_rule:
                    # Special display for groupby
                    if r["rule_id"] == "groupby":
                        params = r["parameters"]
                        group_cols = ", ".join(params.get("group_columns", []))
                        aggs = ", ".join([f"{col}_{func}" for col, func in params.get("aggregations", {}).items()])
                        st.text(f"{i+1}. {rule_name}")
                        st.caption(f"   Agrupar: {group_cols} | Agregaciones: {aggs}")
                    else:
                        st.text(f"{i+1}. {rule_name} : {r['parameters'].get('column', '')}")
                with col_delete:
                    if st.button("", key=f"del_{i}", help="Eliminar regla", icon=":material/delete:"):
                        remove_rule_from_table(selected_table, i, df)
                        st.rerun()
            
            st.write("")
            if st.button("Limpiar todas", type="tertiary", use_container_width=True, icon=":material/clear_all:"):
                clear_rules_for_table(selected_table, df)
                st.rerun()
        else:
            st.caption("No hay reglas aplicadas")
    
    st.divider()
    
    # Preview: Before y After lado a lado
    st.subheader("Preview")
    
    col_before, col_after = st.columns(2)
    
    with col_before:
        st.markdown("**BEFORE**")
        st.dataframe(df.head(15), use_container_width=True, height=350)
    
    with col_after:
        st.markdown("**AFTER**")
        applied_rules = get_applied_rules(selected_table)
        if applied_rules:
            st.dataframe(st.session_state.active_dataframe.head(15), use_container_width=True, height=350)
        else:
            st.caption("Añade reglas para ver el preview")
                
    st.divider()
    
    # Botones de acción
    col_save, col_continue = st.columns([1, 1])
    
    with col_save:
        if st.button("Guardar cambios", type="primary", use_container_width=True, icon=":material/save:"):
            applied_rules = get_applied_rules(selected_table)
            operations = rules_to_operations(applied_rules)
            payload = {"table_name": selected_table, "operations": operations}
            _, success = post("silver", "apply", payload)
            if success:
                st.success("Cambios guardados correctamente en la capa Silver")
            else:
                st.warning("API no disponible. Los cambios no se han guardado.")
    
    with col_continue:
        if st.button("Continuar a Gold", use_container_width=True, icon=":material/arrow_forward:"):
            st.switch_page("pages/3_Gold.py")
    
    st.divider()
    
    # Mostrar tablas Silver existentes
    st.subheader("Tablas Silver existentes")
    
    silver_tables, silver_mock = fetch("silver", "tables")
    
    if silver_mock and not tables_mock:
        st.info("Modo de prueba: API Silver no disponible")
    
    if not silver_tables:
        st.caption("No hay tablas Silver creadas aún")
    else:
        with st.container(height=350):
            for table in silver_tables:
                with st.container(border=True):
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                    with col1:
                        st.write(f"**{table['name']}**")
                    with col2:
                        st.caption(f"{table['rows']:,} filas")
                    with col3:
                        if st.button("Ver", key=f"btn_ver_silver_{table['name']}", use_container_width=True):
                            show_table_detail(table['name'], layer="silver")
                    with col4:
                        if st.button("Eliminar", key=f"btn_del_silver_{table['name']}", use_container_width=True, disabled=silver_mock, type="primary"):
                            confirm_delete(table['name'], layer="silver")
            
if __name__ == "__main__":
    show()