from unittest.mock import patch
from fastapi.testclient import TestClient

import io

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

def test_bronze_upload():
    csv_content = b"id,name,email\n1,John,john@example.com\n2,Jane,jane@example.com"
    csv_file = ("customers.csv", io.BytesIO(csv_content), "text/csv")
    
    response = client.post("/bronze/upload/", files={"files": csv_file})
    assert response.status_code == 201
    
def test_bronze_upload_multiple():
    files = [
        ("files", ("customers.csv", io.BytesIO(b"id,name\n1,John"), "text/csv")),
        ("files", ("orders.csv", io.BytesIO(b"order_id,amount\n100,50.0"), "text/csv"))
    ]
    
    response = client.post("/bronze/upload/", files=files)
    assert response.status_code == 201
    
def test_bronze_upload_not_csv_returns_400():
    csv_content = b"id,name,email\n1,John,john@example.com\n2,Jane,jane@example.com"
    txt_file = ("customers.txt", io.BytesIO(csv_content), "text/plain")
    
    response = client.post("/bronze/upload/", files={"files": txt_file})
    assert response.status_code == 400