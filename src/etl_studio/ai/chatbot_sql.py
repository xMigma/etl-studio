import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq

from etl_studio.ai.chatbot import (
    obtener_esquema,
    es_pregunta_esquema,
    responder_pregunta_esquema,
    generar_sql,
    ejecutar_sql,
    explicar_resultados,
)

load_dotenv()

# Variables globales
client = None
schema_text = None


def inicializar():
    """Inicializa el agente chatbot"""
    global client, schema_text
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("No encuentro GROQ_API_KEY en el .env")
    
    client = Groq(api_key=api_key)
    schema_text = obtener_esquema()


def chat(pregunta: str) -> dict:
    """Metodo principal para el chatbot"""
    if client is None:
        inicializar()
    
    if es_pregunta_esquema(pregunta):
        respuesta = responder_pregunta_esquema(pregunta, client, schema_text)
        return {
            "sql": "",
            "resultados": pd.DataFrame(),
            "error": None,
            "respuesta": respuesta,
            "tipo": "esquema",
        }
    
    sql, error = generar_sql(pregunta, client, schema_text)
    
    if error:
        return {
            "sql": "",
            "resultados": pd.DataFrame(),
            "error": error,
            "respuesta": "",
            "tipo": "datos",
        }
    
    resultados, is_dangerous, error = ejecutar_sql(sql)

    if is_dangerous:
        return {
            "pregunta": pregunta,
            "sql": sql,
            "resultados": pd.DataFrame(),
            "requiere_confirmacion": True,
            "respuesta": "",
            "tipo": "datos",
            "error": error,
        }
    
    if error:
        return {
            "sql": sql,
            "resultados": pd.DataFrame(),
            "error": error,
            "respuesta": "",
            "tipo": "datos",
        }
    
    respuesta = explicar_resultados(pregunta, sql, resultados, client)
    
    return {
        "sql": sql,
        "resultados": resultados,
        "error": None,
        "respuesta": respuesta,
        "tipo": "datos",
    }

