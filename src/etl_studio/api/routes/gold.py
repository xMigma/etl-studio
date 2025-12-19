from fastapi import APIRouter, HTTPException, status
from fastapi.responses import Response
from etl_studio.api.schemas.gold import GoldJoinRequest
from etl_studio.etl.gold import join_tables

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
            preview=preview
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


@router_gold.post("/join/")
def join_gold_tables(request: GoldJoinRequest):
    """Preview a join operation and return as CSV file."""
    result_df = _execute_join(request, preview=True)
    return Response(content=result_df.to_csv(index=False), media_type="text/csv")


@router_gold.post("/apply/")
def apply_gold_join(request: GoldJoinRequest):
    """Apply a join operation and save the result to gold database."""
    _execute_join(request, preview=False)
