from unittest.mock import patch
from fastapi.testclient import TestClient
import pandas as pd
import io

from etl_studio.api.main import app

client = TestClient(app)

# Constantes reutilizables
SAMPLE_CSV_CONTENT = "id,name,age\n1,John,30\n2,Jane,25"


def test_get_available_rules():
    response = client.get("/silver/rules/")
    assert response.status_code == 200
    data = response.json()
    assert "rules" in data
    assert isinstance(data["rules"], dict)


@patch("etl_studio.api.routes.silver.dispatch_operations")
def test_process_preview_silver(mock_dispatch):
    df = pd.read_csv(io.StringIO(SAMPLE_CSV_CONTENT))
    mock_dispatch.return_value = df

    request_data = {
        "table_name": "customers",
        "operations": [
            {"operation": "drop_nulls", "params": {"column": "age"}}
        ]
    }

    response = client.post("/silver/preview/", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    mock_dispatch.assert_called_once()
    assert mock_dispatch.call_args[1]["preview"] is True


@patch("etl_studio.api.routes.silver.dispatch_operations")
def test_process_preview_silver_multiple_operations(mock_dispatch):
    df = pd.read_csv(io.StringIO(SAMPLE_CSV_CONTENT))
    mock_dispatch.return_value = df

    request_data = {
        "table_name": "customers",
        "operations": [
            {"operation": "drop_nulls", "params": {"column": "age"}},
            {"operation": "rename_column", "params": {"column": "name", "new_name": "full_name"}}
        ]
    }

    response = client.post("/silver/preview/", json=request_data)
    assert response.status_code == 200
    mock_dispatch.assert_called_once()


@patch("etl_studio.api.routes.silver.dispatch_operations")
def test_process_preview_silver_error(mock_dispatch):
    mock_dispatch.side_effect = Exception("Processing error")

    request_data = {
        "table_name": "customers",
        "operations": [{"operation": "drop_nulls"}]
    }

    response = client.post("/silver/preview/", json=request_data)
    assert response.status_code == 500


@patch("etl_studio.api.routes.silver.dispatch_operations")
def test_apply_operations_silver(mock_dispatch):
    df = pd.read_csv(io.StringIO(SAMPLE_CSV_CONTENT))
    mock_dispatch.return_value = df

    request_data = {
        "table_name": "customers",
        "operations": [
            {"operation": "drop_nulls", "params": {"column": "age"}}
        ]
    }

    response = client.post("/silver/apply/", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    mock_dispatch.assert_called_once()
    assert mock_dispatch.call_args[1]["preview"] is False


@patch("etl_studio.api.routes.silver.dispatch_operations")
def test_apply_operations_silver_error(mock_dispatch):
    mock_dispatch.side_effect = Exception("Application error")

    request_data = {
        "table_name": "customers",
        "operations": [{"operation": "drop_nulls"}]
    }

    response = client.post("/silver/apply/", json=request_data)
    assert response.status_code == 500


def test_preview_invalid_request_returns_422():
    response = client.post("/silver/preview/", json={})
    assert response.status_code == 422


def test_apply_invalid_request_returns_422():
    response = client.post("/silver/apply/", json={})
    assert response.status_code == 422
