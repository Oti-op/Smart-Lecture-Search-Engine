from fastapi import Request

from backend.services.embedder import Embedder
from backend.services.transcriber import Transcriber


def get_embedder(request: Request) -> Embedder:
    """Return the Embedder singleton stored on app.state."""
    return request.app.state.embedder


def get_transcriber(request: Request) -> Transcriber:
    """Return the Transcriber singleton stored on app.state."""
    return request.app.state.transcriber
