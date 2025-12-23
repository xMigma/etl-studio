from unittest.mock import patch
from fastapi.testclient import TestClient
import pandas as pd
import io

from etl_studio.api.main import app

client = TestClient(app)

SAMPLE_CSV_CONTENT = "id,name,amount\n1,Product A,100.0\n2,Product B,200.0"


def create_sample_dataframe():
    """Create a sample dataframe for testing."""
    return pd.read_csv(io.StringIO(SAMPLE_CSV_CONTENT))


@patch("etl_studio.api.routes.gold.get_gold_tables_info")
def test_list_gold_tables(mock_get_tables):
    expected_tables = [
        {"name": "customers_orders", "rows": 150},
        {"name": "products_sales", "rows": 300},
    ]
    mock_get_tables.return_value = expected_tables
    
    response = client.get("/gold/tables/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["name"] == "customers_orders"
    assert data[0]["rows"] == 150
    assert data[1]["name"] == "products_sales"
    assert data[1]["rows"] == 300
    mock_get_tables.assert_called_once()


@patch("etl_studio.api.routes.gold.get_gold_tables_info")
def test_list_gold_tables_empty(mock_get_tables):
    mock_get_tables.return_value = []
    
    response = client.get("/gold/tables/")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 0


@patch("etl_studio.api.routes.gold.get_gold_tables_info")
def test_list_gold_tables_internal_error(mock_get_tables):
    mock_get_tables.side_effect = Exception("Database error")
    
    response = client.get("/gold/tables/")
    assert response.status_code == 500


@patch("etl_studio.api.routes.gold.join_tables")
def test_join_gold_tables_preview(mock_join):
    df = create_sample_dataframe()
    mock_join.return_value = df
    
    request_data = {
        "left_table": "customers",
        "right_table": "orders",
        "left_source": "silver",
        "right_source": "silver",
        "output_table_name": "customers_orders",
        "config": {
            "left_key": "id",
            "right_key": "customer_id",
            "join_type": "inner"
        }
    }
    
    response = client.post("/gold/join/", json=request_data)
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    mock_join.assert_called_once_with(
        left_table="customers",
        right_table="orders",
        left_source="silver",
        right_source="silver",
        left_key="id",
        right_key="customer_id",
        join_type="inner",
        preview=True,
        output_table_name="customers_orders"
    )


@patch("etl_studio.api.routes.gold.join_tables")
def test_join_gold_tables_left_join(mock_join):
    df = create_sample_dataframe()
    mock_join.return_value = df
    
    request_data = {
        "left_table": "customers",
        "right_table": "orders",
        "left_source": "silver",
        "right_source": "silver",
        "output_table_name": "customers_orders",
        "config": {
            "left_key": "id",
            "right_key": "customer_id",
            "join_type": "left"
        }
    }
    
    response = client.post("/gold/join/", json=request_data)
    assert response.status_code == 200
    mock_join.assert_called_once()
    assert mock_join.call_args[1]["join_type"] == "left"


@patch("etl_studio.api.routes.gold.join_tables")
def test_join_gold_tables_validation_error(mock_join):
    mock_join.side_effect = ValueError("Invalid join key")
    
    request_data = {
        "left_table": "customers",
        "right_table": "orders",
        "left_source": "silver",
        "right_source": "silver",
        "output_table_name": "customers_orders",
        "config": {
            "left_key": "invalid_key",
            "right_key": "customer_id",
            "join_type": "inner"
        }
    }
    
    response = client.post("/gold/join/", json=request_data)
    assert response.status_code == 400
    assert "Validation error" in response.json()["detail"]


@patch("etl_studio.api.routes.gold.join_tables")
def test_join_gold_tables_internal_error(mock_join):
    mock_join.side_effect = Exception("Database error")
    
    request_data = {
        "left_table": "customers",
        "right_table": "orders",
        "left_source": "silver",
        "right_source": "silver",
        "output_table_name": "customers_orders",
        "config": {
            "left_key": "id",
            "right_key": "customer_id",
            "join_type": "inner"
        }
    }
    
    response = client.post("/gold/join/", json=request_data)
    assert response.status_code == 500


def test_join_gold_tables_invalid_request():
    response = client.post("/gold/join/", json={})
    assert response.status_code == 422


@patch("etl_studio.api.routes.gold.join_tables")
def test_apply_gold_join(mock_join):
    df = create_sample_dataframe()
    mock_join.return_value = df
    
    request_data = {
        "left_table": "customers",
        "right_table": "orders",
        "left_source": "silver",
        "right_source": "silver",
        "output_table_name": "customers_orders",
        "config": {
            "left_key": "id",
            "right_key": "customer_id",
            "join_type": "inner"
        }
    }
    
    response = client.post("/gold/apply/", json=request_data)
    assert response.status_code == 200
    mock_join.assert_called_once_with(
        left_table="customers",
        right_table="orders",
        left_source="silver",
        right_source="silver",
        left_key="id",
        right_key="customer_id",
        join_type="inner",
        preview=False,
        output_table_name="customers_orders"
    )


@patch("etl_studio.api.routes.gold.join_tables")
def test_apply_gold_join_validation_error(mock_join):
    mock_join.side_effect = ValueError("Invalid table name")
    
    request_data = {
        "left_table": "invalid_table",
        "right_table": "orders",
        "left_source": "silver",
        "right_source": "silver",
        "output_table_name": "customers_orders",
        "config": {
            "left_key": "id",
            "right_key": "customer_id",
            "join_type": "inner"
        }
    }
    
    response = client.post("/gold/apply/", json=request_data)
    assert response.status_code == 400


@patch("etl_studio.api.routes.gold.join_tables")
def test_apply_gold_join_internal_error(mock_join):
    mock_join.side_effect = Exception("Database error")
    
    request_data = {
        "left_table": "customers",
        "right_table": "orders",
        "left_source": "silver",
        "right_source": "silver",
        "output_table_name": "customers_orders",
        "config": {
            "left_key": "id",
            "right_key": "customer_id",
            "join_type": "inner"
        }
    }
    
    response = client.post("/gold/apply/", json=request_data)
    assert response.status_code == 500


def test_apply_gold_join_invalid_request():
    response = client.post("/gold/apply/", json={})
    assert response.status_code == 422


@patch("etl_studio.api.routes.gold.get_table")
def test_get_gold_table_content(mock_get_table):
    mock_get_table.return_value = SAMPLE_CSV_CONTENT
    
    response = client.get("/gold/tables/customers_orders/")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    assert response.text == SAMPLE_CSV_CONTENT
    mock_get_table.assert_called_once_with("customers_orders", preview=False)


@patch("etl_studio.api.routes.gold.get_table")
def test_get_gold_table_content_preview(mock_get_table):
    mock_get_table.return_value = SAMPLE_CSV_CONTENT
    
    response = client.get("/gold/tables/customers_orders/?preview=true")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv; charset=utf-8"
    mock_get_table.assert_called_once_with("customers_orders", preview=True)


@patch("etl_studio.api.routes.gold.get_table")
def test_get_gold_table_content_internal_error(mock_get_table):
    mock_get_table.side_effect = Exception("Table not found")
    
    response = client.get("/gold/tables/nonexistent_table/")
    assert response.status_code == 500


@patch("etl_studio.api.routes.gold.delete_table")
def test_delete_gold_table(mock_delete_table):
    mock_delete_table.return_value = True
    
    response = client.delete("/gold/tables/customers_orders")
    assert response.status_code == 200
    mock_delete_table.assert_called_once_with("customers_orders")


@patch("etl_studio.api.routes.gold.delete_table")
def test_delete_gold_table_not_found(mock_delete_table):
    mock_delete_table.return_value = False
    
    response = client.delete("/gold/tables/nonexistent_table")
    assert response.status_code == 404
    mock_delete_table.assert_called_once_with("nonexistent_table")


@patch("etl_studio.api.routes.gold.delete_table")
def test_delete_gold_table_internal_error(mock_delete_table):
    mock_delete_table.side_effect = Exception("Database error")
    
    response = client.delete("/gold/tables/customers_orders")
    assert response.status_code == 500
    mock_delete_table.assert_called_once_with("customers_orders")
