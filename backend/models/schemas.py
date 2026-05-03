from pydantic import BaseModel, Field


# --- Upload ---

class UploadResponse(BaseModel):
    lecture_id: str
    duration: float
    chunk_count: int
    message: str


# --- Lecture listing ---

class LectureListItem(BaseModel):
    lecture_id: str
    indexed_at: str  # ISO-8601 UTC datetime string


class LectureListResponse(BaseModel):
    lectures: list[LectureListItem]


# --- Search ---

class SearchRequest(BaseModel):
    lecture_id: str
    query: str = Field(min_length=3, max_length=300)
    top_k: int = Field(default=5, ge=1, le=10)


class SearchResult(BaseModel):
    timestamp: str       # MM:SS of chunk start
    end_timestamp: str   # MM:SS of chunk end
    text: str
    score: float
    chunk_id: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
