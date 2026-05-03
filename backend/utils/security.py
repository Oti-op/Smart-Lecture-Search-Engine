import os
import re
import threading
import time
import uuid
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from fastapi import HTTPException, UploadFile

load_dotenv()

_ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    ext.strip().lower()
    for ext in os.getenv("ALLOWED_EXTENSIONS", "mp3,mp4,wav,m4a,webm").split(",")
    if ext.strip()
)

_MAX_FILE_SIZE_BYTES: int = int(os.getenv("MAX_FILE_SIZE_MB", "100")) * 1024 * 1024

# Authoritative mapping: extension -> valid MIME type prefixes.
# A prefix match is used so codec variants (e.g. video/mp4, audio/mp4) are accepted.
_EXTENSION_MIME_MAP: dict[str, tuple[str, ...]] = {
    "mp3":  ("audio/mpeg", "audio/mp3"),
    "mp4":  ("video/mp4", "audio/mp4"),
    "wav":  ("audio/wav", "audio/x-wav", "audio/wave"),
    "m4a":  ("audio/mp4", "audio/x-m4a", "audio/aac"),
    "webm": ("video/webm", "audio/webm"),
}

_UNSAFE_CHARS_RE = re.compile(r"[^\w.\-]")

# ── Rate limiting ──────────────────────────────────────────────────────────────
# Sliding 60-second window per IP, stored as a list of monotonic timestamps.
# A threading.Lock keeps the dict mutation safe in uvicorn's threaded worker model.
_rate_store: dict[str, list[float]] = defaultdict(list)
_rate_lock = threading.Lock()


def check_rate_limit(ip: str, limit: int = 10) -> None:
    """Raise HTTP 429 if ip has exceeded limit requests in the last 60 seconds.

    Args:
        ip:    Client IP address string.
        limit: Maximum requests allowed per 60-second sliding window.

    Raises:
        HTTPException: 429 with a Retry-After header when the limit is exceeded.
    """
    now = time.monotonic()
    cutoff = now - 60.0
    with _rate_lock:
        window = [t for t in _rate_store[ip] if t > cutoff]
        if len(window) >= limit:
            raise HTTPException(
                status_code=429,
                headers={"Retry-After": "60"},
                detail="Rate limit exceeded. Please wait before sending another request.",
            )
        window.append(now)
        _rate_store[ip] = window


# ── File validation ────────────────────────────────────────────────────────────

async def validate_file(file: UploadFile) -> None:
    """Validate extension, MIME type, and file size of an uploaded file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lstrip(".").lower()

    if ext not in _ALLOWED_EXTENSIONS:
        allowed = ", ".join(sorted(_ALLOWED_EXTENSIONS))
        raise HTTPException(
            status_code=400,
            detail=f"File type '.{ext}' is not allowed. Accepted: {allowed}.",
        )

    content_type = (file.content_type or "").lower().split(";")[0].strip()
    expected_types = _EXTENSION_MIME_MAP.get(ext, ())
    if expected_types and not any(content_type.startswith(t) for t in expected_types):
        raise HTTPException(
            status_code=400,
            detail=(
                f"MIME type '{content_type}' does not match extension '.{ext}'. "
                "Possible file spoofing attempt."
            ),
        )

    # Read the entire body to measure size, then seek back so callers can re-read.
    body = await file.read()
    await file.seek(0)

    if len(body) > _MAX_FILE_SIZE_BYTES:
        limit_mb = _MAX_FILE_SIZE_BYTES // (1024 * 1024)
        actual_mb = len(body) / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File size {actual_mb:.1f} MB exceeds the {limit_mb} MB limit.",
        )

    if len(body) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")


# ── Filename / query sanitisation ──────────────────────────────────────────────

def sanitize_filename(filename: str) -> str:
    """Return a safe, lowercase filename with path traversal removed."""
    # Isolate the stem and extension before any manipulation.
    path = Path(filename)
    stem = path.stem
    ext = path.suffix  # includes the leading dot, e.g. ".mp3"

    # Remove path traversal sequences explicitly, then strip leading separators.
    stem = stem.replace("..", "").replace("/", "").replace("\\", "")
    stem = stem.lstrip("/\\")

    # Replace whitespace and any remaining non-word characters with underscores.
    stem = _UNSAFE_CHARS_RE.sub("_", stem)

    # Collapse consecutive underscores and strip edge underscores for readability.
    stem = re.sub(r"_+", "_", stem).strip("_")

    # Rebuild and enforce length limit (extension kept outside the cap).
    stem = stem[:100]
    if not stem:
        stem = "upload"

    return (stem + ext).lower()


def sanitize_query(query: str) -> str:
    """Strip and validate a search query string."""
    cleaned = query.strip()

    if len(cleaned) < 3:
        raise HTTPException(
            status_code=400,
            detail="Search query must be at least 3 characters long.",
        )

    if len(cleaned) > 300:
        raise HTTPException(
            status_code=400,
            detail="Search query must not exceed 300 characters.",
        )

    return cleaned


def generate_lecture_id() -> str:
    """Return a new UUID4 string for identifying a lecture."""
    return str(uuid.uuid4())
