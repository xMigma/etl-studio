"""Streamlit entry point for the ETL Studio platform."""

from __future__ import annotations

import streamlit as st

from etl_studio.app.pages import (
    chatbot_page,
    cleaning_page,
    gold_page,
    ingest_page,
    model_page,
)

PAGES = {
    "Ingest - Bronze": ingest_page,
    "Cleaning - Silver": cleaning_page,
    "Integration - Gold": gold_page,
    "ML Model": model_page,
    "SQL Chatbot": chatbot_page,
}


def main() -> None:
    """Main Streamlit entry point that routes to the selected page."""

    st.set_page_config(page_title="ETL Studio", layout="wide")
    st.title("ETL Studio")

    selection = st.sidebar.radio("Navigate", list(PAGES.keys()))
    page = PAGES[selection]
    page.show()


if __name__ == "__main__":
    main()
