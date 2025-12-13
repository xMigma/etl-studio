from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from etl_studio.ai import chatbot_sql

app = FastAPI(title="SQL Chatbot API")

class ChatRequest(BaseModel):
    pregunta: str

@app.on_event("startup")
async def startup_event():
    chatbot_sql.inicializar()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: ChatRequest):
    try:
        response = chatbot_sql.chat(request.pregunta)
        
        return {
            "sql": response.get("sql", ""),
            "respuesta": response.get("respuesta", ""),
            "error": response.get("error"),
            "tipo": response.get("tipo", "datos")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/schema")
async def get_schema():
    if chatbot_sql.schema_text is None:
        chatbot_sql.inicializar()
    return {"schema": chatbot_sql.schema_text}
