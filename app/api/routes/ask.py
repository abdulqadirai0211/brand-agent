from app.api.schemas import AskRequest, AskResponse, Citation, TraceStep
from app.dependencies import get_ask_graph
from app.llm.service import LLMUnconfiguredError, llm_error_types
from fastapi import APIRouter, Depends, HTTPException

router = APIRouter(tags=["ask"])


@router.post(
    "/tenants/{tenant_id}/ask",
    response_model=AskResponse,
    status_code=200,
    summary="Ask a question against a tenant's corpus",
)
async def ask_question(
    tenant_id: str,
    request: AskRequest,
    graph: object = Depends(get_ask_graph),
) -> AskResponse:
    try:
        result = await graph.ainvoke(
            {
                "tenant_id": tenant_id,
                "question": request.question,
                "active_query": request.question,
                "retrieved_chunks": [],
                "relevant_chunks": [],
                "retry_count": 0,
                "answer": "",
                "found_in_corpus": False,
                "trace": [],
            }
        )
    except LLMUnconfiguredError as exc:
        raise HTTPException(
            status_code=503,
            detail={"error": "llm_unconfigured", "message": str(exc)},
        ) from exc
    except llm_error_types() as exc:
        raise HTTPException(
            status_code=502,
            detail={"error": "llm_unavailable", "message": str(exc)},
        ) from exc
    grounded = result["found_in_corpus"]
    cited_chunks = result.get("relevant_chunks", []) if grounded else []
    return AskResponse(
        answer=result["answer"],
        found_in_corpus=grounded,
        citations=[
            Citation(
                source=doc.metadata["source"],
                doc_id=doc.metadata["doc_id"],
                chunk_index=doc.metadata["chunk_index"],
                brand=doc.metadata["brand"],
                chunk_text=doc.page_content,
            )
            for doc in cited_chunks
        ],
        trace=[
            TraceStep(node=step["node"], detail=step["detail"])
            for step in result.get("trace", [])
        ],
    )