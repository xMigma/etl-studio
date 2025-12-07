# ETL Studio

[![Tests](https://github.com/xMigma/etl-studio/actions/workflows/tests.yml/badge.svg)](https://github.com/xMigma/etl-studio/actions/workflows/tests.yml)

Lightweight Streamlit skeleton for a collaborative data platform. This repo only
contains scaffolding so each team member can focus on their layer (ETL, ML, or
UX) without stepping on each other's toes.

## Project layout

```
etl-studio/
├─ src/
│  └─ etl_studio/      # Main package
│     ├─ app/          # Streamlit entrypoint + pages
│     ├─ etl/          # Bronze/Silver/Gold helper modules
│     └─ ai/           # ML training + SQL chatbot helpers
├─ test/               # Pytest suite
├─ data/               # Storage zones (bronze/silver/gold)
├─ models/             # Persisted ML artifacts
├─ docs/               # Documentation
├─ test-reports/       # Test coverage and reports
└─ extra_requirements/ # Optional dependencies
```

## Getting started

**Install the project in editable mode** (this eliminates the need for `sys.path` hacks):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or if you want dev dependencies (pytest, etc.):

```bash
pip install -e ".[dev]"
```

### Run the Streamlit workspace

```bash
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
