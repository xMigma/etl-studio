"""Streamlit page for Bronze layer ingestion workflows."""

from __future__ import annotations

import streamlit as st

from etl_studio.app import setup_page
from etl_studio.app.components import show_table_detail_dialog
from etl_studio.app.data import fetch, delete, post

setup_page("Bronze · ETL Studio")


@st.dialog("Detalle de Tabla", width="large")
def show_table_detail(table_name: str) -> None:
    """Display table details in a dialog."""
    show_table_detail_dialog(table_name, layer="bronze")


def show() -> None:
    """Render the ingestion (Bronze) workspace."""

    st.header("Ingest · Bronze")
    
    # Inicializar el contador de upload en session_state
    if "upload_key" not in st.session_state:
        st.session_state.upload_key = 0
    
    st.subheader("Subir archivos")
    uploaded_files = st.file_uploader(
        "Arrastra o selecciona archivos CSV",
        type=["csv"],
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.upload_key}"
    )
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} archivo(s) seleccionado(s). Revisa la selección y pulsa 'Subir archivos' para confirmar.")
        if st.button("Subir archivos"):
            files = [(file.name, file.getvalue()) for file in uploaded_files]
            _, success = post("bronze", "upload", files=files, timeout=30)
            if not success:
                st.warning("API no disponible. Los archivos no se pudieron subir.")
            # Incrementar el key para forzar un nuevo widget limpio
            st.session_state.upload_key += 1
            st.rerun()
    
    st.divider()
    
    st.subheader("Tablas disponibles")
    
    tables, is_mock = fetch("bronze", "tables")
    
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
                            success, message = delete("bronze", "tables", table['name'])
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
