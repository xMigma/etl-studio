"""Streamlit page for Gold layer integration workflows."""

from __future__ import annotations

import streamlit as st

from etl_studio.etl import gold


def show() -> None:
    """Render the integration (Gold) workspace."""

    st.header("Integration · Gold")
    st.write("Placeholder for curated datasets, business rules, and lineage charts.")
    st.info(
        "TODO: orchestrate gold.create_gold_tables and show resulting dataset previews."
    )

    st.caption(f"Integration helper: {gold.create_gold_tables.__name__}")


if __name__ == "__main__":
    show()
