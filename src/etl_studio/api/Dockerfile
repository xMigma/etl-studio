FROM python:3.9

WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements
COPY pyproject.toml setup.py ./
COPY src/ ./src/

# Instalar el paquete
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Exponer puerto
EXPOSE 80

# Ejecutar la API
CMD ["uvicorn", "etl_studio.api.main:app", "--host", "0.0.0.0", "--port", "80", "--reload"]
