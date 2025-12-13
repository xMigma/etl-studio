from pydantic import BaseModel


class BronzeUploadResponse(BaseModel):
    filename: str
    status: str


class BronzeTableName(BaseModel):
    name: str
    rows: int