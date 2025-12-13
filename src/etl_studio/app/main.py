"""Streamlit entry point for the ETL Studio platform."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno al inicio de la aplicación
load_dotenv()


from etl_studio.app import setup_page


def main() -> None:
    """Main Streamlit landing page for the ETL Studio workspace."""

    setup_page("ETL Studio")
    
    # Hero Section
    st.markdown(
        """
        <div style="text-align: center; padding: 2rem 0 1rem 0;">
            <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem;">
                <span style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                    ETL Studio
                </span>
            </h1>
            <p style="font-size: 1.3rem; color: #6b7280; margin-bottom: 2rem;">
                Plataforma unificada para ingeniería de datos, analítica y machine learning
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.divider()
    
    # Cards de navegación
    st.markdown("## :compass: Explora el Workspace")
    st.markdown("")
    
    # Fila 1: Ingeniería de Datos
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container(border=True):
            st.markdown("### Bronze Layer")
            st.markdown("**Ingesta de datos**")
            st.caption("Carga y visualiza datos en bruto desde múltiples fuentes.")
            st.page_link("pages/1_Bronze.py", label="Ir a Bronze →", use_container_width=True)
    
    with col2:
        with st.container(border=True):
            st.markdown("### Silver Layer")
            st.markdown("**Limpieza y transformación**")
            st.caption("Procesa, limpia y estandariza tus datasets.")
            st.page_link("pages/2_Silver.py", label="Ir a Silver →", use_container_width=True)
    
    with col3:
        with st.container(border=True):
            st.markdown("### Gold Layer")
            st.markdown("**Analítica y reportes**")
            st.caption("Datos agregados listos para consumo y visualización.")
            st.page_link("pages/3_Gold.py", label="Ir a Gold →", use_container_width=True)
    
    st.markdown("")
    
    # Fila 2: IA y Chatbot
    col4, col5, col6 = st.columns(3)
    
    with col4:
        with st.container(border=True):
            st.markdown("### Modelo Predictivo")
            st.markdown("**Machine Learning**")
            st.caption("Entrena, evalúa y despliega modelos de ML.")
            st.page_link("pages/4_Modelo_Predictivo.py", label="Ir a Modelo Predictivo →", use_container_width=True)
    
    with col5:
        with st.container(border=True):
            st.markdown("### Chatbot SQL")
            st.markdown("**Consultas en lenguaje natural**")
            st.caption("Pregunta a tus datos usando IA conversacional.")
            st.page_link("pages/5_Chatbot.py", label="Ir a Chatbot →", use_container_width=True)
    
    with col6:
        with st.container(border=True):
            st.markdown("### :material/info: Acerca de")
            st.markdown("**ETL Studio v0.1**")
            st.caption("Proyecto para flujos de datos colaborativos.")
            st.link_button("Ver en GitHub", "https://github.com/xMigma/etl-studio", use_container_width=True)
    
    st.divider()
    
    # Footer con métricas o info adicional
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0; color: #9ca3af;">
            <p>Selecciona una tarjeta o usa la barra lateral para navegar</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
