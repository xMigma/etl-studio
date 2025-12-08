FROM etl-studio-base:latest

# Copiar datos de bronze para desarrollo
COPY data/ ./data/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/etl_studio/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]