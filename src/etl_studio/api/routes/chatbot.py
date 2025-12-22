from fastapi import APIRouter, HTTPException

from etl_studio.ai import chatbot_sql
from etl_studio.api.schemas.chatbot import ChatRequest

router_chatbot = APIRouter(prefix="/chatbot", tags=["chatbot"])


@router_chatbot.get("/health")
async def health():
    return {"status": "ok"}


@router_chatbot.post("/chat")
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


@router_chatbot.get("/schema")
async def get_schema():
    if chatbot_sql.schema_text is None:
        chatbot_sql.inicializar()
    return {"schema": chatbot_sql.schema_text}
