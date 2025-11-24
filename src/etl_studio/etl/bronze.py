"""Bronze layer ingestion helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_csv_to_bronze(file_path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load a raw CSV file into the bronze zone and capture metadata."""

    # TODO: Implement real ingestion logic (validation, persistence, metadata capture).
    df = pd.DataFrame()
    metadata = {"rows": len(df), "columns": list(df.columns), "source": str(file_path)}
    print(f"[Bronze] Ingest placeholder for {file_path}")
    return df, metadata
