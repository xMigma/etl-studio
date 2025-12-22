# ETL Studio

**Build & Testing**

[![Tests](https://github.com/xMigma/etl-studio/actions/workflows/tests.yml/badge.svg)](https://github.com/xMigma/etl-studio/actions/workflows/tests.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/xMigma/etl-studio)

---
**Technology Stack**

[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-336791?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)

Lightweight Streamlit skeleton for a collaborative data platform. This repo only
contains scaffolding so each team member can focus on their layer (ETL, ML, or
UX) without stepping on each other's toes.

![ETL Studio Interface](docs/screenshot.png)

## Getting started

### Run with Docker Compose (Recommended)

Start all services (PostgreSQL, API, and Streamlit) with a single command:

```bash
docker-compose up --build
```

This will start:
- **PostgreSQL** on port 5432
- **MLflow Server** on port 5000
- **FastAPI** on port 80
- **Streamlit** on port 8501



Access the applications:

Streamlit UI: [http://localhost:8501](http://localhost:8501)
MLflow UI: [http://localhost:5000](http://localhost:5000)
FastAPI Docs: [http://localhost:80/docs](http://localhost:80/docs)

To stop all services:

```bash
docker-compose down
```

To stop and clean up volumes (reset database):

```bash
docker-compose down -v
```

### Local Development

For local development without Docker:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
streamlit run src/etl_studio/app/main.py
```

### Run the tests

```bash
pytest
```
