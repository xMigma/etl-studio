"""Streamlit page for Silver layer data cleaning workflows."""

from __future__ import annotations

import pandas as pd
import streamlit as st

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

    st.header("Cleaning · Silver")
    
    # Inicializar estado
    if "selected_rule" not in st.session_state:
        st.session_state.selected_rule = None
    if "test_rule" not in st.session_state:
        st.session_state.test_rule = None
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
    
    # Layout de 3 columnas
    col_rules, col_editor, col_preview = st.columns([1, 1.2, 1.8])
    
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
                
                col_test, col_add = st.columns(2)
                with col_test:
                    if st.button("Probar", type="secondary", use_container_width=True):
                        st.session_state.test_rule = {
                            "rule_id": rule["id"],
                            "column": column,
                            "value": value
                        }
                        st.rerun()
                
                with col_add:
                    if st.button("Añadir", type="primary", use_container_width=True):
                        add_rule_to_table(selected_table, rule["id"], column, value)
                        st.session_state.test_rule = None
                        st.rerun()
        else:
            st.caption("Selecciona una regla para configurarla")
    
    with col_preview:
        st.subheader("Preview")
        
        # Mostrar reglas aplicadas con opción de eliminar
        applied_rules = get_applied_rules(selected_table)
        if applied_rules:
            st.caption("**Reglas aplicadas:**")
            for i, r in enumerate(applied_rules):
                rule_name = next((rule["name"] for rule in AVAILABLE_RULES if rule["id"] == r["rule_id"]), r["rule_id"])
                col_rule, col_delete = st.columns([4, 1])
                with col_rule:
                    st.text(f"{i+1}. {rule_name} → {r['column']}")
                with col_delete:
                    if st.button("", key=f"del_{i}", help="Eliminar regla", icon=":material/delete:"):
                        remove_rule_from_table(selected_table, i)
                        st.rerun()
            
            if st.button("Limpiar todas", type="tertiary", use_container_width=True, icon=":material/clear_all:"):
                clear_rules_for_table(selected_table)
                st.rerun()
            
            st.divider()
        
        tab_before, tab_after = st.tabs(["BEFORE", "AFTER"])
        
        with tab_before:
            st.dataframe(df.head(10), use_container_width=True, height=300)
        
        with tab_after:
            # Aplicar todas las reglas guardadas
            df_after = apply_all_rules(df, applied_rules)
            
            # Si hay regla de prueba, aplicarla también
            if st.session_state.test_rule:
                test = st.session_state.test_rule
                try:
                    df_after = apply_rule(df_after, test["rule_id"], test["column"], test["value"])
                    st.caption("*(Incluyendo regla en prueba)*")
                except Exception as e:
                    st.error(f"Error al aplicar regla: {e}")
            
            st.dataframe(df_after.head(10), use_container_width=True, height=300)
                
    st.divider()
    st.button("Guardar cambios", type="primary", use_container_width=True)


if __name__ == "__main__":
    show()
