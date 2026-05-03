# Smart Lecture Search Engine

A semantic search engine for audio and video lecture recordings. Upload any recording
(MP3, MP4, WAV, M4A, or WebM), and the engine transcribes it with OpenAI Whisper,
splits the transcript into overlapping chunks, embeds each chunk with
`sentence-transformers`, and stores the result in a FAISS vector index. You can then
search across the recording using natural-language queries — finding relevant segments
by meaning rather than exact keyword match.

## Prerequisites

- **Python 3.9+**
- **ffmpeg** — required by Whisper to decode audio from video containers

  ```bash
  # macOS
  brew install ffmpeg

  # Ubuntu / Debian
  sudo apt install ffmpeg

  # Windows (winget)
  winget install ffmpeg
  ```

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd lecture-search

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env to adjust MAX_FILE_SIZE_MB, ALLOWED_EXTENSIONS, or CORS_ORIGINS
```

## Run

```bash
uvicorn backend.main:app --reload
```

Then open <http://127.0.0.1:8000> in your browser. The backend serves the
frontend automatically — no separate file server or port needed.

> **First run:** Whisper downloads the `base` model (~150 MB) and
> `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB) on first use.
> Both are cached locally and not re-downloaded on subsequent starts.

## API docs

Interactive Swagger UI is available at <http://localhost:8000/docs> while the server
is running. ReDoc is at <http://localhost:8000/redoc>.

## Environment variables

| Variable              | Default                  | Description                          |
|-----------------------|--------------------------|--------------------------------------|
| `MAX_FILE_SIZE_MB`    | `100`                    | Maximum upload size in megabytes     |
| `ALLOWED_EXTENSIONS`  | `mp3,mp4,wav,m4a,webm`   | Comma-separated list of allowed file types |
| `CORS_ORIGINS`        | `http://localhost:3000`  | Comma-separated allowed CORS origins (not needed when using the built-in static serving) |
| `TRUSTED_HOSTS`       | *(unset)*                | Comma-separated trusted hostnames; leave unset in development |

## Project layout

```
lecture-search/
  backend/
    main.py               # FastAPI app, middleware, lifespan
    dependencies.py       # Singleton DI getters (Embedder, Transcriber)
    routers/
      upload.py           # POST /api/lectures/upload, GET /api/lectures
      search.py           # POST /api/search
    services/
      transcriber.py      # Whisper transcription
      embedder.py         # sentence-transformers dense embedding
      indexer.py          # FAISS index build / load / search
    models/
      schemas.py          # Pydantic request and response models
    data/
      transcripts/        # Stored transcript JSON files (git-ignored)
      indexes/            # Stored FAISS index files (git-ignored)
    utils/
      file_utils.py       # Chunking, timestamp formatting, JSON I/O
      security.py         # File validation, sanitisation, rate limiting
  frontend/
    index.html            # Self-contained single-page UI
```
