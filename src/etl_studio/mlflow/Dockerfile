FROM python:3.10-slim

WORKDIR /mlflow

# Install curl for health checks and MLflow dependencies
RUN apt-get update && \
    apt-get install -y curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --no-cache-dir mlflow psycopg2-binary

# Expose MLflow port
EXPOSE 5000

# Default command (will be overridden by docker-compose)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
