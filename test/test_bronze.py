from unittest.mock import patch
from fastapi.testclient import TestClient

from etl_studio.api.main import app

client = TestClient(app)


@patch("etl_studio.api.routes.bronze.get_bronze_table_names")
def test_list_bronze_tables(mock_get_tables):
    mock_get_tables.return_value = [
        {"name": "customers", "rows": 100},
        {"name": "orders", "rows": 200},
    ]
    
    response = client.get("/bronze/tables/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["name"] == "customers"
    assert data[0]["rows"] == 100
    assert data[1]["name"] == "orders"
    assert data[1]["rows"] == 200
    
    mock_get_tables.assert_called_once()


@patch("etl_studio.api.routes.bronze.get_bronze_table_names")
def test_list_bronze_tables_empty(mock_get_tables):
    mock_get_tables.return_value = []
    
    response = client.get("/bronze/tables/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0