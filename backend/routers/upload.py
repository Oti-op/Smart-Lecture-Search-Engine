import os
import tempfile
from datetime import datetime, timezone
from typing import Annotated

import aiofiles
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile

from backend.dependencies import get_embedder, get_transcriber
from backend.models.schemas import LectureListItem, LectureListResponse, UploadResponse
from backend.services.embedder import Embedder
from backend.services.indexer import LectureIndex
from backend.services.transcriber import Transcriber
from backend.utils.file_utils import chunk_transcript, save_transcript
from backend.utils.security import generate_lecture_id, sanitize_filename, validate_file

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRANSCRIPTS_DIR = os.path.join(_BACKEND_DIR, "data", "transcripts")
_INDEXES_DIR = os.path.join(_BACKEND_DIR, "data", "indexes")

router = APIRouter(prefix="/api/lectures", tags=["lectures"])


def _remove_file(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_lecture(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    transcriber: Annotated[Transcriber, Depends(get_transcriber)],
    embedder: Annotated[Embedder, Depends(get_embedder)],
) -> UploadResponse:
    """Transcribe, chunk, embed, and index an uploaded audio/video lecture."""
    await validate_file(file)

    lecture_id = generate_lecture_id()
    safe_name = sanitize_filename(file.filename or "upload")
    ext = os.path.splitext(safe_name)[1]

    fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)

    try:
        async with aiofiles.open(temp_path, "wb") as tmp:
            await tmp.write(await file.read())

        transcription = transcriber.transcribe(temp_path)
        segments = transcription["segments"]
        chunks = chunk_transcript(segments)

        if not chunks:
            raise HTTPException(
                status_code=422,
                detail="No speech detected in the uploaded file.",
            )

        index = LectureIndex(lecture_id=lecture_id, data_dir=_INDEXES_DIR)
        index.build(chunks=chunks, embedder=embedder)

        save_transcript(
            lecture_id=lecture_id,
            data={
                "lecture_id": lecture_id,
                "full_text": transcription["full_text"],
                "segments": segments,
                "chunks": chunks,
            },
            data_dir=_TRANSCRIPTS_DIR,
        )

        duration = round(segments[-1]["end"], 2) if segments else 0.0

    except HTTPException:
        _remove_file(temp_path)
        raise
    except FileNotFoundError as exc:
        _remove_file(temp_path)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        _remove_file(temp_path)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    background_tasks.add_task(_remove_file, temp_path)

    return UploadResponse(
        lecture_id=lecture_id,
        duration=duration,
        chunk_count=len(chunks),
        message="Lecture indexed successfully",
    )


@router.get("", response_model=LectureListResponse)
async def list_lectures() -> LectureListResponse:
    """Return all indexed lectures sorted by most recently indexed."""
    if not os.path.exists(_INDEXES_DIR):
        return LectureListResponse(lectures=[])

    lectures: list[LectureListItem] = []
    for fname in os.listdir(_INDEXES_DIR):
        if not fname.endswith(".faiss"):
            continue
        lecture_id = fname.removesuffix(".faiss")
        mtime = os.path.getmtime(os.path.join(_INDEXES_DIR, fname))
        indexed_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
        lectures.append(LectureListItem(lecture_id=lecture_id, indexed_at=indexed_at))

    lectures.sort(key=lambda item: item.indexed_at, reverse=True)
    return LectureListResponse(lectures=lectures)
