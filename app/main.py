from app.api.routes import ask, health, ingest
from fastapi import FastAPI


def create_app() -> FastAPI:
    application = FastAPI(
        title="brand-qa-agent",
        summary="Multi-tenant brand Q&A agent (LangGraph + hybrid Pinecone retrieval)",
        version="0.1.0",
    )
    application.include_router(health.router)
    application.include_router(ingest.router)
    application.include_router(ask.router)
    return application


app = create_app()