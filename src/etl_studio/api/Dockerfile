FROM etl-studio-base:latest

EXPOSE 80

# Ejecutar la API
CMD ["uvicorn", "etl_studio.api.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
