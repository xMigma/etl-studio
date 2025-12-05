"""Streamlit page for Silver layer data cleaning workflows."""

from __future__ import annotations

import pandas as pd
import requests
import streamlit as st

from etl_studio.app import setup_page
from etl_studio.app.mock_data import MOCK_RULES, apply_mock_rules
from etl_studio.config import API_BASE_URL
from etl_studio.etl.bronze import fetch_tables, fetch_table_csv

setup_page("Silver · ETL Studio")


def fetch_rules() -> tuple[dict, bool]:
    """Fetch cleaning rules from API, fallback to mock on failure."""
    try:
        response = requests.get(f"{API_BASE_URL}/cleaning/rules", timeout=5)
        if response.status_code == 200:
            return response.json(), False
    except requests.exceptions.RequestException:
        pass
    return MOCK_RULES, True


def fetch_preview(table: str, rules: list[dict], df: pd.DataFrame) -> pd.DataFrame:
    """Fetch preview from API, fallback to local processing on failure."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/cleaning/preview",
            json={"table": table, "rules": rules},
            timeout=10
        )
        if response.status_code == 200:
            data = response.json()
            return pd.DataFrame(data["after"])
    except requests.exceptions.RequestException:
        pass
    
    # Fallback: procesar localmente
    return apply_mock_rules(df, rules)


def apply_changes(table: str, rules: list[dict]) -> bool:
    """Apply changes via API, return success status."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/cleaning/apply",
            json={"table": table, "rules": rules},
            timeout=10
        )
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


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


def show() -> None:
    """Render the cleaning (Silver) workspace."""
    setup_page("Cleaning · Silver")

    st.header("Cleaning · Silver")
    
    # Inicializar estado
    if "selected_rule" not in st.session_state:
        st.session_state.selected_rule = None
    if "applied_rules" not in st.session_state:
        st.session_state.applied_rules = {}
    
    tables, tables_mock = fetch_tables()
    available_rules, rules_mock = fetch_rules()
    
    if tables_mock or rules_mock:
        st.info("Modo de prueba: API no disponible")
    
    table_names = [table['name'] for table in tables]
    
    selected_table = st.selectbox(
        "Selecciona la tabla a limpiar:",
        table_names,
    )
    
    if not selected_table:
        return
    
    # Cargar datos de la tabla
    df, _ = fetch_table_csv(selected_table)
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
        success = apply_changes(selected_table, applied_rules)
        if success:
            st.success("Cambios guardados correctamente en la capa Silver")
        else:
            st.warning("API no disponible. Los cambios no se han guardado.")
            
if __name__ == "__main__":
    show()
