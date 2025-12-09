"""API routes for Gold layer operations."""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from pydantic import BaseModel

from etl_studio.postgres.gold import (
    get_table_names_db,
    get_table_content_db,
    save_table_db,
    delete_table_db,
)

router_gold = APIRouter(prefix="/gold", tags=["gold"])


class GoldTableName(BaseModel):
    name: str
    rows: int


class SaveTableRequest(BaseModel):
    name: str
    data: list[dict]


@router_gold.get("/tables/", response_model=list[GoldTableName])
def list_gold_tables():
    """List all tables in the Gold layer."""
    try:
        table_names = get_table_names_db()
        return table_names
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving table names: {str(e)}",
        )


@router_gold.get(
    "/tables/{table_name}",
    response_class=Response,
    responses={200: {"content": {"text/csv": {}}}},
)
def get_gold_table(table_name: str, preview: bool = False):
    """Get content of a specific Gold table as CSV."""
    try:
        csv_content = get_table_content_db(table_name, limit=300 if preview else None)
        return Response(content=csv_content, media_type="text/csv")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving table: {str(e)}",
        )


@router_gold.post("/tables/", status_code=status.HTTP_201_CREATED)
def save_gold_table(request: SaveTableRequest):
    """Save a new table to the Gold layer."""
    try:
        save_table_db(request.name, request.data)
        return {"status": "ok", "name": request.name}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving table: {str(e)}",
        )


@router_gold.delete("/tables/{table_name}", status_code=status.HTTP_200_OK)
def delete_gold_table(table_name: str):
    """Delete a table from the Gold layer."""
    try:
        deleted = delete_table_db(table_name)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Table '{table_name}' not found",
            )
        return {"status": "ok"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting table: {str(e)}",
        )