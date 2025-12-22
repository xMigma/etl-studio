"""FastAPI application."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from etl_studio.api.routes.bronze import router_bronze
from etl_studio.api.routes.silver import router_silver
from etl_studio.api.routes.gold import router_gold
from etl_studio.api.routes.chatbot import router_chatbot
from etl_studio.ai import chatbot_sql


@asynccontextmanager
async def lifespan(app: FastAPI):
	chatbot_sql.inicializar()
	yield

app = FastAPI(title="ETL Studio API", version="0.1.0", lifespan=lifespan)

app.include_router(router_bronze)
app.include_router(router_silver)
app.include_router(router_gold)
app.include_router(router_chatbot)


@app.get("/")
async def root():
    return {"message": "ETL Studio API"}


@app.get("/health")
async def health():
    return {"status": "ok"}
