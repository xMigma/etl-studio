FROM etl-studio-base:latest

WORKDIR /mlflow

# Expose MLflow port
EXPOSE 5000

# Default command (will be overridden by docker-compose)
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
