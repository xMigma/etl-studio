"""FastAPI application."""

from fastapi import FastAPI

from etl_studio.api.routes.bronze import router_bronze
from etl_studio.api.routes.gold import router_gold

app = FastAPI(title="ETL Studio API", version="0.1.0")

app.include_router(router_bronze)
app.include_router(router_gold)


@app.get("/")
async def root():
    return {"message": "ETL Studio API"}


@app.get("/health")
async def health():
    return {"status": "ok"}
