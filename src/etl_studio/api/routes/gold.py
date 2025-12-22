from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from etl_studio.api.schemas.gold import GoldJoinRequest, GoldTableName
from etl_studio.etl.gold import join_tables, get_gold_tables_info, get_table, delete_table

router_gold = APIRouter(prefix="/gold", tags=["gold"])


def _execute_join(request: GoldJoinRequest, preview: bool = False):
    """Internal helper to execute a join operation."""
    try:
        result_df = join_tables(
            left_table=request.left_table,
            right_table=request.right_table,
            left_source=request.left_source,
            right_source=request.right_source,
            left_key=request.config.left_key,
            right_key=request.config.right_key,
            join_type=request.config.join_type,
            preview=preview,
            output_table_name = request.output_table_name
        )
        
        return result_df

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Validation error: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing join: {str(e)}"
        )


@router_gold.post("/join/", status_code=status.HTTP_200_OK)
def join_gold_tables(request: GoldJoinRequest):
    """Preview a join operation and return as CSV file."""
    result_df = _execute_join(request, preview=True)
    return Response(content=result_df.to_csv(index=False), media_type="text/csv")


@router_gold.post("/apply/", status_code=status.HTTP_200_OK)
def apply_gold_join(request: GoldJoinRequest):
    """Apply a join operation and save the result in the gold layer."""
    _execute_join(request, preview=False)

@router_gold.get("/tables/", response_model=list[GoldTableName], status_code=status.HTTP_200_OK)
def get_table_names():
    """Get all table names from the gold schema."""
    try:
        tables_info = get_gold_tables_info()
        return [GoldTableName(name=t["name"], rows=t["rows"]) for t in tables_info]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving table names: {str(e)}"
        )


@router_gold.get("/tables/{table_name}/", status_code=status.HTTP_200_OK,)
def get_table_content(table_name: str, preview: bool = False):
    """Get the content of a table from the gold schema as CSV."""
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


@router_gold.delete("/tables/{table_name}", status_code=status.HTTP_200_OK)
def delete_gold_table(table_name: str):
    """Endpoint to delete a specific table from the gold layer."""
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
    