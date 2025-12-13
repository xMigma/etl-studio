
import os
import pandas as pd
from dotenv import load_dotenv
from groq import Groq
from sqlalchemy import create_engine, inspect, text

load_dotenv()

# Variables globales
client = None
engine = None
schema_text = None

# Inicializa el agente, aqui esta el create engine
def inicializar():
    global client, engine, schema_text
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("No encuentro GROQ_API_KEY en el .env")
    
    client = Groq(api_key=api_key)
    
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    user = os.getenv("POSTGRES_USER", "etl_user")
    password = os.getenv("POSTGRES_PASSWORD", "etl_password")
    database = os.getenv("POSTGRES_DB", "etl_database")
    
    db_url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{database}"
    engine = create_engine(db_url)
    
    schema_text = obtener_esquema()

def obtener_esquema():
    ''' Obtiene el esquema de la bbdd'''
    inspector = inspect(engine)
    schema = "Tablas en la base de datos:\n\n"
    
    for table_name in inspector.get_table_names():
        schema += f"Tabla: {table_name}\n"
        
        columns = inspector.get_columns(table_name)
        for col in columns:
            nullable = "NULL" if col["nullable"] else "NOT NULL"
            schema += f"  - {col['name']} ({col['type']}) {nullable}\n"
        
        schema += "\n"
    
    return schema

def es_pregunta_esquema(pregunta):
    ''' Comprueba si la pregunta es sobre el esquema de la bbdd. Para optimizar rendimiento'''
    palabras_clave = [
        "qué tablas", "cuántas tablas", "qué columnas",
        "estructura", "esquema", "describe", "información sobre"
    ] 
    
    pregunta_lower = pregunta.lower()
    return any(palabra in pregunta_lower for palabra in palabras_clave)

def responder_pregunta_esquema(pregunta):
    '''Responde a las preguntas del esquema sobre la bbdd'''
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

def generar_sql(pregunta):
    '''Genera el SQL en base a la consulta del usuario'''
    prompt = f"""Convierte esta pregunta a SQL de PostgreSQL.
    Esquema de la base de datos: {schema_text}
    Reglas:
    - Solo devuelve el SQL, nada más
    - No uses punto y coma al final
    - Usa sintaxis de PostgreSQL
    Ejemplos:
    Pregunta: "Muéstrame todos los usuarios"
    SQL: SELECT * FROM users
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

def ejecutar_sql(sql):
    '''Ejecuta el sql generado por el llm previamente'''
    bannedWords = ["DROP","TRUNCATE","ALTER", "CREATE"]
    is_dangerous = False

    if any(word in sql for word in bannedWords):
        is_dangerous = True

    if (is_dangerous):
        return pd.DataFrame(), True, None

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            
            if result.returns_rows:
                columns = result.keys()
                rows = result.fetchall()
                data = [dict(zip(columns, row)) for row in rows]
                return pd.DataFrame(data), False, None
            else:
                with engine.connect() as conn:
                    conn.commit()
                    return pd.DataFrame([{"filas_afectadas": result.rowcount}]), False, None

    except Exception as e:
        return pd.DataFrame(), False, f"Error al ejecutar SQL: {str(e)}"

def ejecutar_force_sql(sql):
    '''Metodo para ejecutar consultas peligrosas (borrados, creaciones, alteraciones de tablas sobre todo)'''
    print("DEBUG - EJECUTANDO FORCE SQL")
    try:
        with engine.connect() as conn:
            print("DEBUG - CONEXION A LA BBDD")
            conn = conn.execution_options(isolation_level="AUTOCOMMIT")
            result = conn.execute(text(sql))
            print("DEBUG - EJECUTANDO SQL")
            filas_afectadas = result.rowcount if result.rowcount >= 0 else 0
            return pd.DataFrame([{"filas_afectadas": filas_afectadas}]), False, None

    except Exception as e:
        return pd.DataFrame(), False, f"Error al ejecutar SQL: {str(e)}"

def explicar_resultados(pregunta, sql, resultados):
    '''Metodo para mostrar los resultados de la consulta SQL'''
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
        
    except Exception as e:
        if resultados.empty:
            return "No encontré nada."
        else:
            return f"Encontré {len(resultados)} resultado(s)."

def chat(pregunta):
    '''Metodo principal para el chatbot''' 
    if client is None:
        inicializar()
    
    if es_pregunta_esquema(pregunta):
        respuesta = responder_pregunta_esquema(pregunta)
        return {
            "sql": "",
            "resultados": pd.DataFrame(),
            "error": None,
            "respuesta": respuesta,
            "tipo": "esquema",
        }
    
    sql, error = generar_sql(pregunta)
    
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
            "pregunta" : pregunta,
            "sql" : sql,
            "resultados" : pd.DataFrame(),
            "requiere_confirmacion" : True,
            "respuesta" : "",
            "tipo" : "datos",
            "error" : error,
        }
    
    if error:
        return {
            "sql": sql,
            "resultados": pd.DataFrame(),
            "error": error,
            "respuesta": "",
            "tipo": "datos",
        }
    
    respuesta = explicar_resultados(pregunta, sql, resultados)
    
    return {
        "sql": sql,
        "resultados": resultados,
        "error": None,
        "respuesta": respuesta,
        "tipo": "datos",
    }
