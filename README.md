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
- **FastAPI** on port 80
- **Streamlit** on port 8501

Access the Streamlit app at [http://localhost:8501](http://localhost:8501)

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
