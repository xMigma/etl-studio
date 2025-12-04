from fastapi import APIRouter, UploadFile, File, HTTPException, status

from etl_studio.etl.bronze import load_csv_to_bronze
from etl_studio.api.schemas.bronze import BronzeUploadResponse

router_bronze = APIRouter(prefix="/bronze", tags=["bronze"])


@router_bronze.post("/upload/", response_model=list[BronzeUploadResponse], status_code=status.HTTP_201_CREATED)
async def upload_bronze_files(files: list[UploadFile] = File(...)):
    """Endpoint to upload multiple CSV files to the bronze layer."""
    results = []
    
    for file in files:
        if not file.filename.endswith(".csv"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Only CSV files are allowed: {file.filename}"
            )
    
    try:
        for file in files:
            content = await file.read()
            load_csv_to_bronze(file.filename, content)
            
            results.append(BronzeUploadResponse(
                filename=file.filename,
                status="uploaded"
            ))
        
        return results
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing file: {str(e)}"
        )