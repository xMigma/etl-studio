"""Gold layer database operations."""

from __future__ import annotations
from typing import Optional
import pandas as pd
from sqlalchemy import text
from etl_studio.postgres.postgres import get_engine


def _clean_table_name(table_name: str) -> str:
    """Adapt table name to valid SQL identifier."""
    return "".join(c if c.isalnum() or c == "_" else "_" for c in table_name.lower())
