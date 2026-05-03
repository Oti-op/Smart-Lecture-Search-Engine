import os
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request

from backend.dependencies import get_embedder
from backend.models.schemas import SearchRequest, SearchResponse, SearchResult
from backend.services.embedder import Embedder
from backend.services.indexer import LectureIndex
from backend.utils.file_utils import format_timestamp
from backend.utils.security import check_rate_limit, sanitize_query

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_INDEXES_DIR = os.path.join(_BACKEND_DIR, "data", "indexes")

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search_lecture(
    request: Request,
    body: SearchRequest,
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> SearchResponse:
    """Semantic search over a single indexed lecture."""
    ip = request.client.host if request.client else "unknown"
    check_rate_limit(ip, limit=30)

    # sanitize_query enforces the 3–300 character bounds before the embedder runs.
    query = sanitize_query(body.query)

    index = LectureIndex(lecture_id=body.lecture_id, data_dir=_INDEXES_DIR)
    try:
        index.load()
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"No index found for lecture '{body.lecture_id}'.",
        )

    query_vector = embedder.embed_query(query)

    try:
        raw_results = index.search(query_vector, top_k=body.top_k)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    results = [
        SearchResult(
            timestamp=format_timestamp(r["start"]),
            end_timestamp=format_timestamp(r["end"]),
            text=r["text"],
            score=r["score"],
            chunk_id=r["chunk_id"],
        )
        for r in raw_results
    ]

    return SearchResponse(query=query, results=results)
