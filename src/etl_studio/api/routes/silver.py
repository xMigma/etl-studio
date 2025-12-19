from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response

from etl_studio.api.constants import SILVER_OPERATIONS
from etl_studio.api.schemas.silver import SilverPreviewRequest, SilverTableName
from etl_studio.etl.silver import dispatch_operations, get_table, delete_table, get_silver_tables_info

router_silver = APIRouter(prefix="/silver", tags=["silver"])


@router_silver.get("/rules/")
def get_available_rules():
    """Returns all available cleaning rules."""
    return {"rules": SILVER_OPERATIONS}


@router_silver.post("/preview/")
def process_preview_silver(request: SilverPreviewRequest):
    try:
        operations = [op.model_dump(exclude_none=True) for op in request.operations]
        df_preview = dispatch_operations(request.table_name, operations, preview=True)
        return Response(content=df_preview.to_csv(index=False), media_type="text/csv")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing preview: {str(e)}"
        )


@router_silver.post("/apply/")
def apply_operations_silver(request: SilverPreviewRequest):
    try:
        operations = [op.model_dump(exclude_none=True) for op in request.operations]
        df_result = dispatch_operations(request.table_name, operations, preview=False)
        return Response(content=df_result.to_csv(index=False), media_type="text/csv")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error applying operations: {str(e)}"
        )
    

@router_silver.get("/tables/", response_model=list[SilverTableName], status_code=status.HTTP_200_OK)
def get_table_names():
    """Get all table names from the silver schema."""
    try:
        tables_info = get_silver_tables_info()
        return [SilverTableName(name=t["name"], rows=t["rows"]) for t in tables_info]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving table names: {str(e)}"
        )


@router_silver.get("/tables/{table_name}/", status_code=status.HTTP_200_OK,)
def get_table_content(table_name: str, preview: bool = False):
    """Get the content of a table from the silver schema as CSV."""
    try:
        csv_content = get_table(table_name, preview=preview)
        
        headers = {}
        if not preview:
            headers["Content-Disposition"] = f"attachment; filename={table_name}.csv"
        
        return Response(
            content=csv_content,
            media_type="text/csv",
            headers=headers
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving table content: {str(e)}"
        )


@router_silver.delete("/tables/{table_name}", status_code=status.HTTP_200_OK)
def delete_silver_table(table_name: str):
    """Endpoint to delete a specific table from the silver layer."""
    try:
        deleted = delete_table(table_name)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting table: {str(e)}"
        )
    
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Table '{table_name}' not found"
        )
    
