from pydantic import BaseModel


class BronzeUploadResponse(BaseModel):
    filename: str
    status: str