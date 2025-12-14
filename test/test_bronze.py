from unittest.mock import patch
from fastapi.testclient import TestClient
import io
import pytest

from etl_studio.api.main import app

client = TestClient(app)

# Constantes reutilizables
SAMPLE_CSV_CONTENT = b"id,name,email\n1,John,john@example.com\n2,Jane,jane@example.com"


# Helper functions
def create_csv_file(filename: str, content: bytes = SAMPLE_CSV_CONTENT) -> tuple:
    """Create a mock CSV file for testing."""
    return (filename, io.BytesIO(content), "text/csv")


def create_file(filename: str, content: bytes, mime_type: str) -> tuple:
    """Create a mock file with custom MIME type."""
    return (filename, io.BytesIO(content), mime_type)


def assert_table_list_response(response, expected_tables: list[dict]):
    """Verify table list response format and content."""
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == len(expected_tables)
    
    for i, expected in enumerate(expected_tables):
        assert data[i]["name"] == expected["name"]
        assert data[i]["rows"] == expected["rows"]


# Tests GET
@patch("etl_studio.api.routes.bronze.get_bronze_table_names")
def test_list_bronze_tables(mock_get_tables):
    expected_tables = [
        {"name": "customers", "rows": 100},
        {"name": "orders", "rows": 200},
    ]
    mock_get_tables.return_value = expected_tables
    
    response = client.get("/bronze/tables/")
    assert_table_list_response(response, expected_tables)
    mock_get_tables.assert_called_once()


@patch("etl_studio.api.routes.bronze.get_bronze_table_names")
def test_list_bronze_tables_empty(mock_get_tables):
    mock_get_tables.return_value = []
    
    response = client.get("/bronze/tables/")
    assert_table_list_response(response, [])


# Tests POST usando los helpers
def test_bronze_upload():
    csv_file = create_csv_file("customers.csv")
    response = client.post("/bronze/upload/", files={"files": csv_file})
    assert response.status_code == 201


def test_bronze_upload_multiple():
    files = [
        ("files", create_csv_file("customers.csv", b"id,name\n1,John")),
        ("files", create_csv_file("orders.csv", b"order_id,amount\n100,50.0"))
    ]
    response = client.post("/bronze/upload/", files=files)
    assert response.status_code == 201


def test_bronze_upload_not_csv_returns_400():
    txt_file = create_file("customers.txt", SAMPLE_CSV_CONTENT, "text/plain")
    response = client.post("/bronze/upload/", files={"files": txt_file})
    assert response.status_code == 400


def test_bronze_upload_no_files_returns_422():
    response = client.post("/bronze/upload/", files={})
    assert response.status_code == 422


@patch("etl_studio.api.routes.bronze.load_csv_to_bronze")
def test_bronze_upload_internal_error(mock_load_csv):
    mock_load_csv.side_effect = Exception("Database error")
    csv_file = create_csv_file("customers.csv")
    response = client.post("/bronze/upload/", files={"files": csv_file})
    assert response.status_code == 500