from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response

from etl_studio.api.constants import SILVER_OPERATIONS
from etl_studio.api.schemas.silver import SilverPreviewRequest
from etl_studio.etl.silver import dispatch_operations

router_silver = APIRouter(prefix="/silver", tags=["silver"])


@router_silver.get("/rules/")
def get_available_rules():
    """Returns all available cleaning rules."""
    return {"rules": SILVER_OPERATIONS}


@router_silver.post("/preview/")
def process_preview_silver(request: SilverPreviewRequest):
    """Endpoint to preview the result of applying cleaning operations to a preview table."""
    try:
        df_preview = dispatch_operations(request.table_name, request.operations, preview=True)
        return Response(content=df_preview.to_csv(index=False), media_type="text/csv")

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing preview: {str(e)}"
        )
    
@router_silver.post("/apply/")
def apply_operations_silver(request: SilverPreviewRequest):
    """Endpoint to apply cleaning operations to a table and save it to the silver layer."""
    try:
        dispatch_operations(request.table_name, request.operations, preview=False)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error applying operations: {str(e)}"
        )