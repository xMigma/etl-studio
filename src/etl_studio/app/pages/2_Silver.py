"""Streamlit page for Silver layer data cleaning workflows."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from etl_studio.app import setup_page
from etl_studio.app.components import show_table_detail_dialog
from etl_studio.app.data import fetch, fetch_table_csv, post
from etl_studio.app.mock_data import apply_mock_rules

setup_page("Silver · ETL Studio")


def fetch_preview(table: str, rules: list[dict], df: pd.DataFrame) -> pd.DataFrame:
    """Fetch preview from API, fallback to local processing on failure."""
    data, success = post("silver", "preview", {"table": table, "rules": rules})
    if success and data:
        return pd.DataFrame(data["after"])
    return apply_mock_rules(df, rules)


def get_applied_rules(table_name: str) -> list[dict]:
    """Get applied rules for a table from session state."""
    if "applied_rules" not in st.session_state:
        st.session_state.applied_rules = {}
    return st.session_state.applied_rules.get(table_name, [])


def add_rule_to_table(table_name: str, rule_id: str, column: str, value: str) -> None:
    """Add a rule to the table's applied rules."""
    if "applied_rules" not in st.session_state:
        st.session_state.applied_rules = {}
    
    if table_name not in st.session_state.applied_rules:
        st.session_state.applied_rules[table_name] = []
        
    table = st.session_state.applied_rules[table_name]
    exists = any(r["rule_id"] == rule_id and r["column"] == column for r in table) 
    
    if not exists:
        st.session_state.applied_rules[table_name].append({
            "rule_id": rule_id,
            "column": column,
            "value": value
        })


def remove_rule_from_table(table_name: str, index: int) -> None:
    """Remove a rule from the table's applied rules by index."""
    if "applied_rules" in st.session_state and table_name in st.session_state.applied_rules:
        if 0 <= index < len(st.session_state.applied_rules[table_name]):
            st.session_state.applied_rules[table_name].pop(index)


def clear_rules_for_table(table_name: str) -> None:
    """Clear all rules for a table."""
    if "applied_rules" in st.session_state and table_name in st.session_state.applied_rules:
        st.session_state.applied_rules[table_name] = []


@st.dialog("Detalle de Tabla", width="large")
def show_table_detail(table_name: str) -> None:
    """Display table details in a dialog."""
    show_table_detail_dialog(table_name, layer="silver")


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
    
    st.divider()
    
    # Layout de 2 columnas para reglas y configuración
    col_rules, col_editor, col_applied = st.columns([1.5, 1.5, 1.5], gap="large")
    
    with col_rules:
        st.subheader("Reglas")
        
        for rule_id, rule_data in available_rules.items():
            is_selected = st.session_state.selected_rule == rule_id
            button_type = "primary" if is_selected else "secondary"
            
            if st.button(
                rule_data['name'],
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
            rule = available_rules.get(rule_id)
            
            if rule:
                column = st.selectbox("Columna:", df.columns.tolist(), key="rule_column")     
                value = ""
                if rule.get("requires_value", False) or rule_id == "fillna":
                    value = st.text_input("Valor de relleno:", key="rule_value")
                
                if st.button("Añadir", type="primary", use_container_width=True, icon=":material/add:"):
                    add_rule_to_table(selected_table, rule_id, column, value)
                    st.rerun()
        else:
            st.caption("Selecciona una regla para configurarla")
    
    with col_applied:
        st.subheader("Reglas aplicadas")
        
        applied_rules = get_applied_rules(selected_table)
        if applied_rules:
            for i, r in enumerate(applied_rules):
                rule_data = available_rules.get(r["rule_id"])
                rule_name = rule_data["name"] if rule_data else r["rule_id"]
                col_rule, col_delete = st.columns([4, 1])
                with col_rule:
                    st.text(f"{i+1}. {rule_name} : {r['column']}")
                with col_delete:
                    if st.button("", key=f"del_{i}", help="Eliminar regla", icon=":material/delete:"):
                        remove_rule_from_table(selected_table, i)
                        st.rerun()
            
            st.write("")
            if st.button("Limpiar todas", type="tertiary", use_container_width=True, icon=":material/clear_all:"):
                clear_rules_for_table(selected_table)
                st.rerun()
        else:
            st.caption("No hay reglas aplicadas")
    
    st.divider()
    
    # Preview: Before y After lado a lado
    st.subheader("Preview")
    
    col_before, col_after = st.columns(2)
    
    df_after = fetch_preview(selected_table, applied_rules, df)
    
    with col_before:
        st.markdown("**BEFORE**")
        st.dataframe(df.head(15), use_container_width=True, height=350)
    
    with col_after:
        st.markdown("**AFTER**")
        st.dataframe(df_after.head(15), use_container_width=True, height=350)
                
    st.divider()
    
    if st.button("Guardar cambios", type="primary", use_container_width=True, icon=":material/save:"):
        _, success = post("silver", "apply", {"table": selected_table, "rules": applied_rules})
        if success:
            st.success("Cambios guardados correctamente en la capa Silver")
        else:
            st.warning("API no disponible. Los cambios no se han guardado.")
            
if __name__ == "__main__":
    show()
