"""Sanity tests for the ETL Studio skeleton."""


def test_math_is_sane() -> None:
    """A trivial placeholder test to keep the suite green."""
    assert 1 + 1 == 2


def test_imports() -> None:
    """Test that all main modules can be imported."""
    from etl_studio import app, etl, ai

    assert app is not None
    assert etl is not None
    assert ai is not None


def test_version() -> None:
    """Test that version is accessible."""
    from etl_studio import __version__

    assert __version__ == "0.1.0"
