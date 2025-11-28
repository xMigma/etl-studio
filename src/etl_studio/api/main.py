"""FastAPI application."""

from fastapi import FastAPI

app = FastAPI(title="ETL Studio API", version="0.1.0")


@app.get("/")
async def root():
    return {"message": "ETL Studio API"}


@app.get("/health")
async def health():
    return {"status": "ok"}


# Incluir routers aquí cuando los crees:
# from .routes import etl, ai
# app.include_router(etl.router)
# app.include_router(ai.router)
