"""Streamlit application package for the ETL Studio platform."""

import streamlit as st


def setup_page(title: str = "ETL Studio") -> None:
    """Configure page settings and global styles."""
    st.set_page_config(page_title=title, layout="wide")
    
    st.markdown("""
        <style>
            .stMainBlockContainer, .stAppViewBlockContainer {
                padding-left: 2rem;
                padding-right: 2rem;
                max-width: 100%;
            }
        </style>
    """, unsafe_allow_html=True)
