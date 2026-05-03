import logging
import os
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

_APP_VERSION = "1.0.0"

_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_TRANSCRIPTS_DIR = os.path.join(_BACKEND_DIR, "data", "transcripts")
_INDEXES_DIR = os.path.join(_BACKEND_DIR, "data", "indexes")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # --- startup ---
    os.makedirs(_TRANSCRIPTS_DIR, exist_ok=True)
    os.makedirs(_INDEXES_DIR, exist_ok=True)

    # Import here so model weights are not loaded until the server starts.
    from backend.services.embedder import Embedder
    from backend.services.transcriber import Transcriber

    app.state.embedder = Embedder()
    app.state.transcriber = Transcriber()

    logger.info("Server ready")
    yield

    # --- shutdown ---
    logger.info("Server shutting down")


def _build_app() -> FastAPI:
    app = FastAPI(
        title="Smart Lecture Search Engine API",
        version=_APP_VERSION,
        lifespan=lifespan,
    )

    # CORS
    cors_origins = [
        origin.strip()
        for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # Routers
    from backend.routers.upload import router as upload_router
    from backend.routers.search import router as search_router

    app.include_router(upload_router)
    app.include_router(search_router)

    # Health check
    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        return {"status": "ok", "version": _APP_VERSION}

    # Global exception handler — never leak stack traces to the client
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception("Unhandled error on %s %s", request.method, request.url.path)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"},
        )

    return app


app = _build_app()
