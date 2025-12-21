"""Streamlit entry point for the ETL Studio platform."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno al inicio de la aplicación
load_dotenv()


from etl_studio.app import setup_page


# Definición de las tarjetas de navegación
NAVIGATION_CARDS = [
    # Fila 1: Capas de datos
    [
        {
            "icon": ":material/database:",
            "title": "Bronze Layer",
            "subtitle": "Ingesta de datos",
            "description": "Carga y visualiza datos en bruto desde múltiples fuentes.",
            "page": "pages/1_Bronze.py",
            "label": "Ir a Bronze",
            "color": "#cd7f32",
        },
        {
            "icon": ":material/filter_alt:",
            "title": "Silver Layer",
            "subtitle": "Limpieza y transformación",
            "description": "Procesa, limpia y estandariza tus datasets.",
            "page": "pages/2_Silver.py",
            "label": "Ir a Silver",
            "color": "#a0a0a0",
        },
        {
            "icon": ":material/star:",
            "title": "Gold Layer",
            "subtitle": "Analítica y reportes",
            "description": "Datos agregados listos para consumo y visualización.",
            "page": "pages/3_Gold.py",
            "label": "Ir a Gold",
            "color": "#d4af37",
        },
    ],
    # Fila 2: IA y herramientas
    [
        {
            "icon": ":material/model_training:",
            "title": "Modelo Predictivo",
            "subtitle": "Machine Learning",
            "description": "Entrena, evalúa y despliega modelos de ML.",
            "page": "pages/4_Modelo_Predictivo.py",
            "label": "Ir a Modelo Predictivo",
            "color": "#667eea",
        },
        {
            "icon": ":material/chat:",
            "title": "Chatbot SQL",
            "subtitle": "Consultas en lenguaje natural",
            "description": "Pregunta a tus datos usando IA conversacional.",
            "page": "pages/5_Chatbot.py",
            "label": "Ir a Chatbot",
            "color": "#11998e",
        },
        {
            "icon": ":material/info:",
            "title": "Acerca de",
            "subtitle": "ETL Studio v0.1",
            "description": "Proyecto para flujos de datos colaborativos.",
            "external_link": "https://github.com/xMigma/etl-studio",
            "label": "Ver en GitHub",
            "color": "#6b7280",
        },
    ],
]


def render_card(card: dict) -> None:
    """Render a navigation card."""
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="
                display: flex;
                flex-direction: column;
                min-height: 120px;
            ">
                <div style="margin-bottom: 0.3rem;">
                    <span style="
                        font-size: 1.3rem;
                        font-weight: 700;
                        color: {card["color"]};
                    ">{card["title"]}</span>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <strong>{card["subtitle"]}</strong>
                </div>
                <div style="
                    flex-grow: 1;
                    color: rgba(150, 150, 150, 0.9);
                    font-size: 0.875rem;
                ">{card["description"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if "page" in card:
            st.page_link(
                card["page"],
                label=card["label"],
                icon=card["icon"],
                use_container_width=True,
            )
        elif "external_link" in card:
            st.link_button(
                card["label"], card["external_link"], use_container_width=True
            )


def main() -> None:
    """Main Streamlit landing page for the ETL Studio workspace."""

    setup_page("ETL Studio")

    # Hero Section
    st.markdown(
        """
        <div style="text-align: center; padding: 3rem 0 2rem 0;">
            <h1 style="font-size: 3.5rem; margin-bottom: 0.5rem; font-weight: 800;">
                <span style="
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                ">
                    ETL Studio
                </span>
            </h1>
            <p style="font-size: 1.3rem; color: #6b7280; margin-bottom: 1rem; max-width: 600px; margin-left: auto; margin-right: auto;">
                Plataforma unificada para ingeniería de datos, analítica y machine learning
            </p>
            <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 1.5rem;">
                <div style="text-align: center;">
                    <span style="font-size: 1.5rem; font-weight: 700; color: #667eea;">3</span>
                    <p style="font-size: 0.85rem; color: #9ca3af; margin: 0;">Capas de datos</p>
                </div>
                <div style="text-align: center;">
                    <span style="font-size: 1.5rem; font-weight: 700; color: #764ba2;">ML</span>
                    <p style="font-size: 0.85rem; color: #9ca3af; margin: 0;">Integrado</p>
                </div>
                <div style="text-align: center;">
                    <span style="font-size: 1.5rem; font-weight: 700; color: #f093fb;">IA</span>
                    <p style="font-size: 0.85rem; color: #9ca3af; margin: 0;">Chatbot SQL</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # Sección de navegación
    st.markdown(
        """
        <h2 style="text-align: center; margin-bottom: 0.5rem;">
            Explora el Workspace
        </h2>
        <p style="text-align: center; color: #6b7280; margin-bottom: 2rem;">
            Selecciona un módulo para comenzar
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Renderizar tarjetas
    for row in NAVIGATION_CARDS:
        cols = st.columns(len(row))
        for col, card in zip(cols, row):
            with col:
                render_card(card)
        st.markdown("")

    st.divider()

    # Footer
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0; color: #9ca3af;">
            <p style="margin: 0;">
                Usa la barra lateral para navegar rápidamente entre módulos
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
