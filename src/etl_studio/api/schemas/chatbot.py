from pydantic import BaseModel


class ChatRequest(BaseModel):
    pregunta: str
