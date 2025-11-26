"""Streamlit page for Bronze layer ingestion workflows."""

from __future__ import annotations

import streamlit as st

from etl_studio.etl import bronze


def show() -> None:
    """Render the ingestion (Bronze) workspace."""

    st.header("Ingest · Bronze")
    st.write(
        "Placeholder for file uploads, schema validations, and ingestion monitoring."
    )
    st.info(
        "TODO: wire upload widgets to bronze.load_csv_to_bronze and surface metadata."
    )

    st.caption(f"Preview helper available: {bronze.load_csv_to_bronze.__name__}")


if __name__ == "__main__":
    show()
