from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DocumentInput(BaseModel):
    url: str | None = None
    text: str | None = None
    brand: str = ""
    source: str | None = None

    @model_validator(mode="after")
    def _validate_document(self) -> "DocumentInput":
        if not self.url and not self.text:
            raise ValueError("document must provide a url or text")
        if self.url and self.text:
            raise ValueError("document must provide either url or text, not both")
        if not self.brand.strip():
            raise ValueError("document.brand is required")
        return self


class IngestRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1)


class DocumentError(BaseModel):
    url: str | None = None
    source: str | None = None
    code: str
    message: str


class IngestResponse(BaseModel):
    documents_stored: int
    chunks_stored: int
    errors: list[DocumentError] = Field(default_factory=list)


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class Citation(BaseModel):
    source: str
    doc_id: str
    chunk_index: int
    brand: str
    chunk_text: str


class TraceStep(BaseModel):
    node: Literal["retrieve", "grade", "rewrite", "answer", "not_in_corpus"]
    detail: str


class AskResponse(BaseModel):
    answer: str
    found_in_corpus: bool
    citations: list[Citation]
    trace: list[TraceStep]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    pinecone: Literal["ok", "error"]
    ollama: Literal["ok", "error"]
    llm: Literal["ok", "error", "unconfigured"]