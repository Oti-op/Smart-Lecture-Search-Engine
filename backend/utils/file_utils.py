import json
import os


def chunk_transcript(segments: list[dict], chunk_size: int = 5) -> list[dict]:
    """Group Whisper segments into overlapping chunks for embedding.

    Consecutive chunks share one segment of overlap so that sentences split
    across a boundary appear in both neighbours, preserving search context.

    Args:
        segments: List of {"start": float, "end": float, "text": str} dicts
                  as returned by Transcriber.transcribe().
        chunk_size: Number of segments per chunk. Must be >= 2 so that a
                    step of (chunk_size - 1) is always positive.

    Returns:
        List of {"start": float, "end": float, "text": str, "chunk_id": int}.
        Returns an empty list when segments is empty.

    Raises:
        ValueError: If chunk_size is less than 2.
    """
    if chunk_size < 2:
        raise ValueError(f"chunk_size must be >= 2, got {chunk_size}.")

    if not segments:
        return []

    step = chunk_size - 1  # overlap of 1 segment
    chunks: list[dict] = []
    chunk_id = 0

    for i in range(0, len(segments), step):
        window = segments[i : i + chunk_size]
        chunks.append(
            {
                "start": float(window[0]["start"]),
                "end": float(window[-1]["end"]),
                "text": " ".join(seg["text"].strip() for seg in window),
                "chunk_id": chunk_id,
            }
        )
        chunk_id += 1

    return chunks


def format_timestamp(seconds: float) -> str:
    """Convert a duration in seconds to a MM:SS string.

    Args:
        seconds: Non-negative float number of seconds.

    Returns:
        Zero-padded string in the form "MM:SS", e.g. 143.5 -> "02:23".
    """
    total_seconds = int(seconds)
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def save_transcript(lecture_id: str, data: dict, data_dir: str) -> None:
    """Persist a transcript dict as JSON to the transcripts directory.

    Args:
        lecture_id: UUID string used as the filename stem.
        data: Transcript dict to serialise (must be JSON-serialisable).
        data_dir: Path to the data/transcripts/ directory.
    """
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, f"{lecture_id}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def load_transcript(lecture_id: str, data_dir: str) -> dict:
    """Load a transcript JSON file from the transcripts directory.

    Args:
        lecture_id: UUID string identifying the lecture.
        data_dir: Path to the data/transcripts/ directory.

    Returns:
        The deserialised transcript dict.

    Raises:
        FileNotFoundError: If no transcript exists for lecture_id.
    """
    path = os.path.join(data_dir, f"{lecture_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Transcript not found for lecture '{lecture_id}'.")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
