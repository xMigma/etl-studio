FROM python:3.9-slim

WORKDIR /app

# Solo instalar curl para el healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml setup.py ./
COPY src/ ./src/
COPY data/ ./data/

RUN pip3 install -e .

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/etl_studio/app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]