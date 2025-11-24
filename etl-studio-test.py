"""Simple test script to verify the package structure."""

import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import etl_studio

    print("etl_studio package imported successfully")

    from etl_studio import app, etl, ai

    print("All submodules accessible")

    print("\nPackage structure test passed!")

except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)
