from app.api.schemas import (
    DocumentError,
    IngestRequest,
    IngestResponse,
)
from app.dependencies import get_ingestion
from app.ingestion.models import IngestDocument
from app.ingestion.service import IngestionService
from fastapi import APIRouter, Depends

router = APIRouter(tags=["ingest"])


@router.post(
    "/tenants/{tenant_id}/ingest",
    response_model=IngestResponse,
    status_code=200,
    summary="Ingest documents for a tenant",
)
async def ingest_documents(
    tenant_id: str,
    request: IngestRequest,
    service: IngestionService = Depends(get_ingestion),
) -> IngestResponse:
    documents = [
        IngestDocument(url=doc.url, text=doc.text, brand=doc.brand, source=doc.source)
        for doc in request.documents
    ]
    result = await service.run(tenant_id, documents)
    return IngestResponse(
        documents_stored=result.documents_stored,
        chunks_stored=result.chunks_stored,
        errors=[
            DocumentError(
                url=error.document.url,
                source=error.document.source,
                code=error.code,
                message=error.message,
            )
            for error in result.errors
        ],
    )