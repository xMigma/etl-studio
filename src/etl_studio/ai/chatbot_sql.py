"""SQL chatbot helpers for Gold data."""

from __future__ import annotations

import pandas as pd


def generate_sql_from_prompt(prompt: str, catalog: dict[str, str]) -> str:
    """Generate a SQL statement from a natural language prompt."""

    # TODO: Integrate LLM and schema grounding logic using `catalog` metadata.
    print(f"[Chatbot] Prompt received: {prompt}")
    return "SELECT * FROM some_table LIMIT 10;"


def run_sql_query(sql: str, gold_path: str) -> pd.DataFrame:
    """Execute the generated SQL against the Gold layer."""

    # TODO: Connect to warehouse / duckdb / sqlite built from gold_path files.
    print(f"[Chatbot] Running SQL: {sql} on {gold_path}")
    return pd.DataFrame()
