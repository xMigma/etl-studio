from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import Response

from etl_studio.api.constants import SILVER_OPERATIONS

router_silver = APIRouter(prefix="/silver", tags=["silver"])


@router_silver.get("/rules/")
def get_available_rules():
    """Returns all available cleaning rules."""
    return {"rules": SILVER_OPERATIONS}
