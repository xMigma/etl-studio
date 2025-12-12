from pydantic import BaseModel
from typing import Any


class OperationParams(BaseModel):
    column: str | None = None
    value: Any | None = None
    new_name: str | None = None

class Operation(BaseModel):
    operation: str
    params: OperationParams | None = None

class SilverPreviewRequest(BaseModel):
    """Request body for silver layer preview."""
    table_name: str 
    operations: list[Operation]