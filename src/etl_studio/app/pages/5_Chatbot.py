"""Streamlit page for the SQL chatbot with natural language interface."""

from __future__ import annotations

import streamlit as st

from etl_studio.app import setup_page
from etl_studio.ai import chatbot_sql

setup_page("Chatbot · ETL Studio")


def show() -> None:
    """Render the SQL chatbot workspace."""
    
    st.header("🤖 SQL Chatbot")
    st.write("Haz preguntas en lenguaje natural y el chatbot generará y ejecutará consultas SQL.")
    
    # Initialize chatbot in session state
    if "chatbot_initialized" not in st.session_state:
        try:
            chatbot_sql.inicializar()
            st.session_state.chatbot_initialized = True
            st.session_state.messages = []
        except ValueError as e:
            st.error(f"❌ Error de configuración: {e}")
            st.info("💡 Asegúrate de tener un archivo `.env` con tu `GROQ_API_KEY`")
            return
        except Exception as e:
            st.error(f"❌ Error al conectar con la base de datos: {e}")
            st.info("💡 Verifica que PostgreSQL esté corriendo con `docker-compose up -d`")
            return
    
    # Chat history
    st.subheader("💬 Conversación")
    
    # Mostrar mensajes anteriores
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Solo mostrar SQL y resultados para consultas de datos
            if message.get("tipo") != "esquema":
                if "sql" in message and message["sql"]:
                    with st.expander("🔍 Ver SQL generado"):
                        st.code(message["sql"], language="sql")
                
                if "resultados" in message and not message["resultados"].empty:
                    st.dataframe(message["resultados"], use_container_width=True)
            
            if "error" in message and message["error"]:
                st.error(f"❌ Error: {message['error']}")
    
    # Chat input
    if prompt := st.chat_input("Escribe tu pregunta aquí..."):
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Procesar consulta
        with st.chat_message("assistant"):
            with st.spinner("Procesando..."):
                response = chatbot_sql.chat(prompt)
                print("DEBUG - RESPONSE:", response)
            
            # Si hay error
            if response["error"]:
                st.error(f"❌ Error: {response['error']}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": "Hubo un error al procesar tu consulta.",
                    "error": response["error"],
                })
            if response.get("requiere_confirmacion") == True:
                print("DEBUG - IS DANGEROUS, entra al if correctamente")
                st.session_state.sql_pendiente = response.get("sql")
                st.session_state.pregunta_pendiente = response.get("pregunta")
                st.session_state.requiere_confirmacion = True
            else:
                # Mostrar respuesta en lenguaje natural
                if response.get("respuesta"):
                    st.markdown(f"💬 **{response['respuesta']}**")
                
                # Si es pregunta de esquema, no mostrar SQL
                if response.get("tipo") == "esquema":
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["respuesta"],
                        "tipo": "esquema",
                    })
                else:
                    # Mostrar SQL (colapsado)
                    if response["sql"]:
                        with st.expander("🔍 Ver SQL generado"):
                            st.code(response["sql"], language="sql")
                    
                    # Mostrar resultados
                    if not response["resultados"].empty:
                        st.success("✅ Consulta ejecutada")
                        st.dataframe(response["resultados"], use_container_width=True)
                        
                        num_filas = len(response["resultados"])
                        st.caption(f"📊 {num_filas} fila(s)")
                    else:
                        if response["sql"] and not response["sql"].upper().startswith("SELECT"):
                            st.success("✅ Operación completada")
                    
                    # Guardar en historial
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.get("respuesta", "Consulta ejecutada"),
                        "sql": response["sql"],
                        "resultados": response["resultados"],
                        "tipo": "datos",
                    })
    if (st.session_state.get("requiere_confirmacion")):
        st.warning("Operaciones de borrado o de modificaciones de tablas requieren confirmacion del usuario")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Confirmar"):
                resultados, _, error = chatbot_sql.ejecutar_force_sql(st.session_state.get("sql_pendiente"))
                if error:
                    st.error(f"Error al ejecutar: {error}")
                else:
                    st.success("Operacion ejecutada correctamente")

                    if not resultados.empty and "filas_afectadas" in resultados.columns:
                        filas = resultados.iloc[0]["filas_afectadas"]
                        st.metric("Filas afectadas", filas)
                    st.session_state.messages.append({
                        "role" : "assistant",
                        "content" : f"Operacion ejecutada {st.session_state.get('pregunta_pendiente')}",
                        "sql" : st.session_state.get("sql_pendiente"),
                        "resultados" : resultados,
                    })
                del st.session_state.requiere_confirmacion
                del st.session_state.sql_pendiente
                del st.session_state.pregunta_pendiente

                st.rerun()
        with col2:
            if st.button("Cancelar"):
                st.info("Operacion cancelada por el usuario")
                st.session_state.messages.append({
                    "role" : "assistant",
                    "content" : "Operacion cancelada por el usuario"
                })

                del st.session_state.requiere_confirmacion
                del st.session_state.sql_pendiente
                del st.session_state.pregunta_pendiente

                st.rerun()
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Limpiar conversación"):
            st.session_state.messages = []
            st.rerun()
    
    # Example queries
    st.divider()
    st.subheader("💡 Ejemplos de consultas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Preguntas sobre el esquema:**
        - "¿Qué tablas hay en la base de datos?"
        - "¿Qué columnas tiene la tabla users?"
        - "Explícame la estructura de etl_jobs"
        - "¿Qué relaciones hay entre las tablas?"
        """)
    
    with col2:
        st.markdown("""
        **Consultas de datos:**
        - "Muéstrame todos los usuarios"
        - "¿Cuántos trabajos ETL hay en estado pending?"
        - "Lista las últimas 5 configuraciones"
        - "Muestra los trabajos ETL con sus creadores"
        """)
    
    with col3:
        st.markdown("""
        **Operaciones de escritura:**
        - "Agrega un usuario con username 'test' y email 'test@example.com'"
        - "Actualiza el estado del trabajo con id 1 a 'completed'"
        - "Elimina los logs más antiguos de 60 días"
        """)


if __name__ == "__main__":
    show()
