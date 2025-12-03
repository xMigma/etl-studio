from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from etl_studio.etl.bronze import load_csv_to_bronze

router_bronze = APIRouter(prefix="/bronze", tags=["bronze"])


@router_bronze.post("/upload/")
async def upload_bronze_file(file: UploadFile = File(...)):
    """Endpoint to upload a file to the bronze layer."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV files are allowed"
        )
    
    try:
        content = await file.read()
        load_csv_to_bronze(file.filename, content)
        
        return JSONResponse(
            status_code=status.HTTP_201_CREATED,
            content={
                "filename": file.filename,
                "status": "uploaded"
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )
    
