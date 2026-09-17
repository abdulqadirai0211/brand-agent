import os
from functools import lru_cache

import certifi
from pydantic_settings import BaseSettings, SettingsConfigDict

os.environ.setdefault("SSL_CERT_FILE", certifi.where())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "brand-qa-agent"
    environment: str = "development"
    log_level: str = "INFO"

    # Extraction (WebBaseLoader)
    web_loader_timeout_ms: int = 30_000
    web_user_agent: str = "brand-qa-agent/0.1"
    min_content_length: int = 200

    # Embeddings (Ollama)
    embedding_model: str = "qwen3-embedding:4b"
    embedding_dimension: int = 1024
    ollama_base_url: str = "http://localhost:11434"

    # Chunking (character-based length via the builtin `len`)
    chunk_size: int = 400
    chunk_overlap: int = 50
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_device: str = "cpu"

    # Retrieval
    hybrid_top_k: int = 10  # candidates fetched before reranking (over-retrieve)
    final_top_k: int = 5  # survivors after reranking (context passed to the LLM)

    # LLM (Groq primary; OpenAI / OpenAI-compatible endpoint optional)
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""  # optional OpenAI-compatible endpoint (e.g. local Ollama /v1)
    llm_structured_output_method: str = "json_schema"  # grader structured-output mode
    temperature: float = 0.0  # chat temperature for grade / rewrite / answer

    # Vector store (Pinecone; serverless index, hybrid dense + BM25 sparse)
    pinecone_api_key: str = ""
    pinecone_index: str = "brand-chunks"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_alpha: float = 0.5  # hybrid dense weight; 1.0 = dense only, 0.0 = sparse only


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()