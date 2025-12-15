# ETL Studio

[![Tests](https://github.com/xMigma/etl-studio/actions/workflows/tests.yml/badge.svg)](https://github.com/xMigma/etl-studio/actions/workflows/tests.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/xMigma/etl-studio)

Lightweight Streamlit skeleton for a collaborative data platform. This repo only
contains scaffolding so each team member can focus on their layer (ETL, ML, or
UX) without stepping on each other's toes.

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

## Next steps

- Flesh out ETL logic in `src/etl_studio/etl/bronze.py`, `src/etl_studio/etl/silver.py`, and `src/etl_studio/etl/gold.py`.
- Implement ML training/prediction logic in `src/etl_studio/ai/model.py`.
- Connect the SQL chatbot to an LLM provider inside `src/etl_studio/ai/chatbot_sql.py`.
- Replace placeholder text in each Streamlit page with interactive widgets.
- Run tests with coverage: `pytest` (reports will be in `test-reports/`)
