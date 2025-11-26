"""Streamlit page for the Gold SQL chatbot."""

from __future__ import annotations

import streamlit as st

from etl_studio.ai import chatbot_sql


def show() -> None:
    """Render the SQL chatbot workspace."""

    st.header("SQL Chatbot")
    st.write("Placeholder for prompt input, generated SQL, and result rendering.")
    st.info(
        "TODO: integrate LLM-backed chatbot to call chatbot_sql.generate_sql_from_prompt "
        "and execute via chatbot_sql.run_sql_query."
    )

    st.caption(
        f"Chat helper: {chatbot_sql.generate_sql_from_prompt.__name__} · Runner: {chatbot_sql.run_sql_query.__name__}"
    )


if __name__ == "__main__":
    show()
