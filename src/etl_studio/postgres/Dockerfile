FROM python:3.9

WORKDIR /code

# Copiar pyproject.toml para instalar dependencias
COPY pyproject.toml /code/pyproject.toml

# Instalar dependencias desde pyproject.toml
# Extraemos las dependencias y las instalamos directamente
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    psycopg2>=2.9.11 \
    python-dotenv>=1.2.1 \
    SQLAlchemy>=2.0.44

# Copiar el script de inicialización
COPY src/postgres/init_db.py /code/init_db.py

# Ejecutar el script de inicialización
CMD ["python", "init_db.py"]
