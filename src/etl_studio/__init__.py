"""ETL Studio - Lightweight Streamlit platform for collaborative data workflows."""

__version__ = "0.1.0"

# Make submodules available
from . import app
from . import etl
from . import ai

__all__ = ["app", "etl", "ai", "__version__"]
