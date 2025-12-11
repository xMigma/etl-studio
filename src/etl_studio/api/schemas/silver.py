from pydantic import BaseModel, Field
from typing import Any, Optional


class OperationParams(BaseModel):
    """Parameters for a cleaning operation."""
    column: Optional[str] = None
    value: Optional[Any] = None
    new_name: Optional[str] = None


class Operation(BaseModel):
    """Single cleaning operation."""
    operation: str = Field(..., description="Name of the operation (fillna, drop_nulls, etc)")
    params: OperationParams = Field(default_factory=OperationParams, description="Parameters for the operation")


class SilverPreviewRequest(BaseModel):
    """Request body for silver layer preview."""
    table_name: str = Field(..., description="Name of the table in bronze layer")
    operations: list[Operation] = Field(..., description="List of cleaning operations to apply")
