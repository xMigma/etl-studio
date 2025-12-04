"""Streamlit page for Silver layer data cleaning workflows."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from etl_studio.app import setup_page
from etl_studio.etl.bronze import fetch_tables, fetch_table_csv


# Reglas disponibles para limpieza
AVAILABLE_RULES = [
    {"id": "fillna", "name": "FillNA", "description": "Rellenar valores nulos"},
    {"id": "trim", "name": "Trim", "description": "Eliminar espacios en blanco"},
    {"id": "lowercase", "name": "Lowercase", "description": "Convertir a minúsculas"},
    {"id": "cast_date", "name": "Cast Date", "description": "Convertir a fecha"},
]


def apply_rule(df: pd.DataFrame, rule_id: str, column: str, value: str) -> pd.DataFrame:
    """Apply a cleaning rule to the dataframe."""
    result = df.copy()
    
    if rule_id == "fillna":
        result[column] = result[column].fillna(value)
    elif rule_id == "trim":
        if result[column].dtype == "object":
            result[column] = result[column].str.strip()
    elif rule_id == "lowercase":
        if result[column].dtype == "object":
            result[column] = result[column].str.lower()
    elif rule_id == "cast_date":
        result[column] = pd.to_datetime(result[column], errors="coerce")
    
    return result


def apply_all_rules(df: pd.DataFrame, rules: list[dict]) -> pd.DataFrame:
    """Apply all rules in order to the dataframe."""
    result = df.copy()
    for rule in rules:
        result = apply_rule(result, rule["rule_id"], rule["column"], rule["value"])
    return result


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
    
    tables, is_mock = fetch_tables()
    
    if is_mock:
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
    col_rules, col_editor, col_applied = st.columns([1, 1.5, 1.5])
    
    with col_rules:
        st.subheader("Reglas")
        
        for rule in AVAILABLE_RULES:
            is_selected = st.session_state.selected_rule == rule["id"]
            button_type = "secondary" if is_selected else "tertiary"
            
            if st.button(
                rule['name'], 
                key=f"btn_{rule['id']}", 
                use_container_width=True,
                type=button_type
            ):
                st.session_state.selected_rule = rule["id"]
                st.session_state.test_rule = None
                st.rerun()
    
    with col_editor:
        st.subheader("Configuración")
        
        if st.session_state.selected_rule:
            rule = next((r for r in AVAILABLE_RULES if r["id"] == st.session_state.selected_rule), None)
            
            if rule:
                st.info(f"**{rule['name']}**\n\n{rule['description']}")
                
                column = st.selectbox("Columna:", df.columns.tolist(), key="rule_column")
                
                value = ""
                if rule["id"] == "fillna":
                    value = st.text_input("Valor de relleno:", key="rule_value")
                
                st.divider()
                
                if st.button("Añadir", type="primary", use_container_width=True, icon=":material/add:"):
                    add_rule_to_table(selected_table, rule["id"], column, value)
                    st.rerun()
        else:
            st.caption("Selecciona una regla para configurarla")
    
    with col_applied:
        st.subheader("Reglas aplicadas")
        
        applied_rules = get_applied_rules(selected_table)
        if applied_rules:
            for i, r in enumerate(applied_rules):
                rule_name = next((rule["name"] for rule in AVAILABLE_RULES if rule["id"] == r["rule_id"]), r["rule_id"])
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
    
    with col_before:
        st.markdown("**BEFORE**")
        st.dataframe(df.head(15), use_container_width=True, height=350)
    
    with col_after:
        st.markdown("**AFTER**")
        # Aplicar todas las reglas guardadas
        df_after = apply_all_rules(df, applied_rules)
        st.dataframe(df_after.head(15), use_container_width=True, height=350)
                
    st.divider()
    st.button("Guardar cambios", type="primary", use_container_width=True, icon=":material/save:")


if __name__ == "__main__":
    show()
