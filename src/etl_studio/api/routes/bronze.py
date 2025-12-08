from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import Response

from etl_studio.etl.bronze import load_csv_to_bronze, get_bronze_table_names, get_table_content, delete_table
from etl_studio.api.schemas.bronze import BronzeUploadResponse, BronzeTableName

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
    
@router_bronze.get("/tables/", response_model=list[BronzeTableName], status_code=status.HTTP_200_OK)
def list_bronze_tables():
    """Endpoint to list all table names in the bronze layer."""
    try:
        table_names = get_bronze_table_names()
        return table_names
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving table names: {str(e)}"
        )
    

@router_bronze.get("/tables/{table_name}",
    response_class=Response,
    responses={
        200: {
            "content": {"text/csv": {}},
            "description": "CSV file download or preview"
        }
    }
)
def download_table_csv(table_name: str, preview: bool = False):
    """Download content of a specific bronze table as CSV file."""
    try:
        csv_content = get_table_content(table_name, limit=300 if preview else None)
        
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
    
@router_bronze.delete("/tables/{table_name}", status_code=status.HTTP_200_OK)
def delete_bronze_table(table_name: str):
    """Endpoint to delete a specific table from the bronze layer."""
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