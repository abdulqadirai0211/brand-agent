import asyncio

import httpx
from app.api.schemas import HealthResponse
from app.config import get_settings
from app.dependencies import get_store
from app.vectorstore.pinecone import PineconeVectorStore
from fastapi import APIRouter, Depends

router = APIRouter(tags=["health"])


async def _pinecone_status(store: PineconeVectorStore) -> str:
    try:
        await asyncio.to_thread(store.count)
    except Exception:
        return "error"
    return "ok"


async def _ollama_status() -> str:
    base_url = get_settings().ollama_base_url
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{base_url}/api/tags")
            response.raise_for_status()
    except Exception:
        return "error"
    return "ok"


async def _llm_status() -> str:
    settings = get_settings()
    if settings.groq_api_key:
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=settings.groq_api_key, timeout=5.0)
            await client.models.list()
        except Exception:
            return "error"
        return "ok"
    if settings.openai_api_key:
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url or None,
                timeout=5.0,
            )
            await client.models.list()
        except Exception:
            return "error"
        return "ok"
    return "unconfigured"


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health(store: PineconeVectorStore = Depends(get_store)) -> HealthResponse:
    pinecone, ollama, llm = await asyncio.gather(
        _pinecone_status(store), _ollama_status(), _llm_status()
    )
    status = "ok" if (pinecone == "ok" and ollama == "ok" and llm in ("ok", "unconfigured")) else "degraded"
    return HealthResponse(status=status, pinecone=pinecone, ollama=ollama, llm=llm)