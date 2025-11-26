"""Streamlit page for Silver layer data cleaning workflows."""

from __future__ import annotations

import streamlit as st

from etl_studio.etl import silver


def show() -> None:
    """Render the cleaning (Silver) workspace."""

    st.header("Cleaning · Silver")
    st.write(
        "Placeholder for transformation recipes, profiling charts, and quality gates."
    )
    st.info("TODO: trigger silver.clean_data and display before/after comparisons.")

    st.caption(f"Transformation helper: {silver.clean_data.__name__}")


if __name__ == "__main__":
    show()
