"""Centralized configuration for ETL Studio."""

from __future__ import annotations

import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
