"""Chatbot SQL logic and utilities."""

import pandas as pd
from groq import Groq
from sqlalchemy import text
from typing import Optional, Tuple

from etl_studio.postgres.postgres import get_engine, get_table_names, get_table_columns


def obtener_esquema():
    """Obtiene el esquema de la bbdd usando SQLAlchemy Inspector"""
    schema = "Tablas en la base de datos:\n\n"
    
    schema_name = "gold"
    tables = get_table_names(schema_name)

    if tables:
        schema += f"Esquema: {schema_name}\n"
        
        for table_name in tables:
            schema += f"Tabla: {table_name}\n"
            
            columns = get_table_columns(table_name, schema_name)
            for col in columns:
                nullable = "NULL" if col["nullable"] else "NOT NULL"
                schema += f"  - {col['name']} ({col['type']}) {nullable}\n"
    
        schema += "\n"
    
    return schema


def es_pregunta_esquema(pregunta: str) -> bool:
    """Comprueba si la pregunta es sobre el esquema de la bbdd. Para optimizar rendimiento"""
    palabras_clave = [
        "qué tablas", "cuántas tablas", "qué columnas",
        "estructura", "esquema", "describe", "información sobre"
    ] 
    
    pregunta_lower = pregunta.lower()
    return any(palabra in pregunta_lower for palabra in palabras_clave)


def responder_pregunta_esquema(pregunta: str, client: Groq, schema_text: str) -> str:
    """Responde a las preguntas del esquema sobre la bbdd"""
    prompt = f"""Eres un asistente de base de datos. 
    Responde esta pregunta sobre el esquema: {schema_text} 
    Pregunta: {pregunta} 
    Responde de forma clara y concisa."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Eres un asistente útil de bases de datos."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500,
        )
        
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def generar_sql(pregunta: str, client: Groq, schema_text: str) -> Tuple[str, Optional[str]]:
    """Genera el SQL en base a la consulta del usuario"""
    prompt = f"""Convierte esta pregunta a SQL de PostgreSQL.
    Esquema de la base de datos: {schema_text}
    Reglas:
    - Solo devuelve el SQL, nada más
    - No uses punto y coma al final
    - Usa sintaxis de PostgreSQL
    - IMPORTANTE: Todas las tablas están en el esquema 'gold', SIEMPRE usa el prefijo gold.nombre_tabla
    Ejemplos:
    Pregunta: "Muéstrame todos los usuarios"
    SQL: SELECT * FROM gold.users
    Pregunta: "Cuántas filas tiene la tabla orders?"
    SQL: SELECT COUNT(*) FROM gold.orders
    Pregunta: {pregunta}
    SQL:"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Eres un experto en SQL."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500,
        )
        
        sql = response.choices[0].message.content.strip()
        
        if sql.startswith("```"):
            sql = sql.split("\n")[1]
        
        sql = sql.strip().rstrip(";")
        
        return sql, None
        
    except Exception as e:
        return "", f"Error al generar SQL: {str(e)}"


def ejecutar_sql(sql: str) -> Tuple[pd.DataFrame, bool, Optional[str]]:
    """Ejecuta el sql generado por el llm previamente"""
    bannedWords = ["DROP", "TRUNCATE", "ALTER", "CREATE"]
    is_dangerous = False

    if any(word in sql for word in bannedWords):
        is_dangerous = True

    if is_dangerous:
        return pd.DataFrame(), True, None

    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            
            if result.returns_rows:
                columns = result.keys()
                rows = result.fetchall()
                data = [dict(zip(columns, row)) for row in rows]
                return pd.DataFrame(data), False, None
            else:
                conn.commit()
                return pd.DataFrame([{"filas_afectadas": result.rowcount}]), False, None

    except Exception as e:
        return pd.DataFrame(), False, f"Error al ejecutar SQL: {str(e)}"


def ejecutar_force_sql(sql: str) -> Tuple[pd.DataFrame, bool, Optional[str]]:
    """Metodo para ejecutar consultas peligrosas (borrados, creaciones, alteraciones de tablas)"""
    try:
        engine = get_engine()
        with engine.connect() as conn:
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            result = conn.execute(text(sql))
            filas_afectadas = result.rowcount if result.rowcount >= 0 else 0
            return pd.DataFrame([{"filas_afectadas": filas_afectadas}]), False, None

    except Exception as e:
        return pd.DataFrame(), False, f"Error al ejecutar SQL: {str(e)}"


def explicar_resultados(pregunta: str, sql: str, resultados: pd.DataFrame, client: Groq) -> str:
    """Metodo para mostrar los resultados de la consulta SQL"""
    if resultados.empty:
        resumen = "No se encontraron resultados."
    else:
        num_filas = len(resultados)
        resumen = f"Se encontraron {num_filas} fila(s).\n"
        if num_filas > 0:
            resumen += f"\nPrimeras filas:\n{resultados.head(3).to_string()}"
    
    prompt = f"""Explica estos resultados de forma natural y amigable.
    Pregunta del usuario: "{pregunta}"
    SQL ejecutado: {sql}
    Resultados: {resumen}
    Da una respuesta breve y clara."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "Explica resultados de forma clara."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300,
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception:
        if resultados.empty:
            return "No encontré nada."
        else:
            return f"Encontré {len(resultados)} resultado(s)."
