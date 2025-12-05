"""Streamlit application package for the ETL Studio platform."""

import streamlit as st


def setup_page(title: str = "ETL Studio") -> None:
    """Configure page settings and global styles."""
    st.set_page_config(page_title=title, layout="wide")
    
    st.markdown("""
        <style>
            /* Contenido principal */
            .stMainBlockContainer, .stAppViewBlockContainer {
                padding-left: 2rem;
                padding-right: 2rem;
                max-width: 100%;
            }
            
            /* ========== SIDEBAR STYLES ========== */
            
            /* Links de navegación con más espacio */
            [data-testid="stSidebarNav"] a {
                padding: 0.75rem 1rem;
                margin: 0.15rem 0.5rem;
                border-radius: 6px;
                font-size: 1.1rem !important;
                transition: background 0.15s ease;
            }
            
            [data-testid="stSidebarNav"] a span {
                font-size: 1.1rem !important;
            }
            
            [data-testid="stSidebarNav"] a:hover {
                background: rgba(128, 128, 128, 0.15);
            }
            
            /* Separador después de Main (primer elemento) */
            [data-testid="stSidebarNav"] li:nth-child(1)::after {
                content: "";
                display: block;
                height: 1px;
                background: rgba(128, 128, 128, 0.3);
                margin: 0.6rem 0.75rem;
            }
            
            /* Separador después de Gold (cuarto elemento) */
            [data-testid="stSidebarNav"] li:nth-child(4)::after {
                content: "";
                display: block;
                height: 1px;
                background: rgba(128, 128, 128, 0.3);
                margin: 0.6rem 0.75rem;
            }
        </style>
    """, unsafe_allow_html=True)
