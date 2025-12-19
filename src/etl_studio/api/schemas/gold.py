from pydantic import BaseModel
from typing import Literal


class JoinConfig(BaseModel):
    """Configuration for joining tables."""
    left_key: str
    right_key: str
    join_type: Literal["inner", "left", "right", "outer"] = "inner"


class GoldJoinRequest(BaseModel):
    """Request body for joining tables in gold layer."""
    left_table: str
    right_table: str
    left_source: Literal["silver", "gold"]
    right_source: Literal["silver", "gold"]
    config: JoinConfig


