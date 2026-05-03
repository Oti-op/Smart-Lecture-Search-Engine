import logging
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.trustedhost import TrustedHostMiddleware

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

    # TrustedHost — only active when TRUSTED_HOSTS is set in the environment.
    # Registered first so it is innermost; invalid-host requests are rejected
    # before reaching any application logic.
    trusted_hosts_env = os.getenv("TRUSTED_HOSTS", "").strip()
    if trusted_hosts_env:
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=[h.strip() for h in trusted_hosts_env.split(",") if h.strip()],
        )

    # CORS — wraps TrustedHost so CORS headers are added to all passed responses.
    cors_origins = [
        o.strip()
        for o in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
        if o.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    # Request logging — registered last so it becomes the outermost layer and
    # captures every request, including those rejected by host/CORS middleware.
    # The request body is intentionally never read here to avoid buffering uploads.
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "%s %s → %d  (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    # Routers
    from backend.routers.search import router as search_router
    from backend.routers.upload import router as upload_router

    app.include_router(upload_router)
    app.include_router(search_router)

    # Health check
    @app.get("/health", tags=["meta"])
    async def health() -> dict:
        return {"status": "ok", "version": _APP_VERSION}

    # Global exception handler — never leak stack traces to the client.
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
