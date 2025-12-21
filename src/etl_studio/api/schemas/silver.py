from pydantic import BaseModel
from typing import Any, Optional


class OperationParams(BaseModel):
    column: Optional[str] = None
    value: Optional[Any] = None
    new_name: Optional[str] = None
    group_columns: Optional[list[str]] = None
    aggregations: Optional[dict[str,str]] = None

class Operation(BaseModel):
    operation: str
    params: Optional[OperationParams] = None

class SilverPreviewRequest(BaseModel):
    """Request body for silver layer preview."""
    table_name: str 
    operations: list[Operation]

class SilverTableName(BaseModel):
    """Model for silver table name and row count."""
    name: str
    rows: int