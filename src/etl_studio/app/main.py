"""Streamlit entry point for the ETL Studio platform."""

from __future__ import annotations

import streamlit as st
from dotenv import load_dotenv

# Cargar variables de entorno al inicio de la aplicación
load_dotenv()


def main() -> None:
    """Main Streamlit landing page for the ETL Studio workspace."""

    st.set_page_config(page_title="ETL Studio", layout="wide")
    st.title("ETL Studio")
    st.subheader("Multipage workspace for ELT, ML, and analytics experiments")
    st.write(
        "Selecciona las distintas capas desde el menú de Streamlit situado en la izquierda. "
        "Cada página vive en `src/etl_studio/app/pages/` y se ejecuta de forma independiente, "
        "por lo que no necesitamos un radio personalizado."
    )
    st.info(
        "Esta página es solo un landing informativo; las vistas Bronze, Silver, Gold, ML y Chatbot "
        "aparecen en la barra lateral nativa de Streamlit."
    )


if __name__ == "__main__":
    main()
