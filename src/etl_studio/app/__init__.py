"""Streamlit application package for the ETL Studio platform."""

import streamlit as st

# CSS global para toda la aplicación (minimalista para evitar parpadeos)
_GLOBAL_CSS = """
<style>
    /* Contenido principal */
    .stMainBlockContainer, .stAppViewBlockContainer {
        padding-left: 2rem;
        padding-right: 2rem;
        max-width: 100%;
    }
    
    /* Sidebar: solo ajustar legibilidad (sin efectos que generen salto visual) */
    [data-testid="stSidebarNav"] a {
        padding: 0.65rem 0.9rem !important;
        margin: 0.2rem 0.6rem !important;
        border-radius: 6px !important;
        font-size: 1.05rem !important;
    }
    
    [data-testid="stSidebarNav"] a span {
        font-size: 1.05rem !important;
    }

    /* Igualar altura de botones en tarjetas */
    .stLinkButton {
        height: 38px;
    }
</style>
"""


def setup_page(title: str = "ETL Studio") -> None:
    """Configure page settings and global styles."""
    st.set_page_config(page_title=title, layout="wide")
    inject_css()


def inject_css() -> None:
    """Inject global CSS styles. Can be called independently."""
    st.markdown(_GLOBAL_CSS, unsafe_allow_html=True)
