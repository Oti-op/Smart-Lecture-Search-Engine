# Smart Lecture Search Engine

Search across lecture audio/video using semantic similarity.

## Stack

- **Backend**: Python, FastAPI, Uvicorn
- **ML**: openai-whisper, sentence-transformers, faiss-cpu
- **Frontend**: Vanilla HTML/JS
- **Storage**: JSON files (prototype)

## Setup

```bash
cd lecture-search
cp .env.example .env
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

## Project Structure

```
lecture-search/
  backend/
    main.py               # FastAPI app entry point
    routers/
      upload.py           # File upload endpoints
      search.py           # Semantic search endpoints
    services/
      transcriber.py      # Whisper transcription
      embedder.py         # Sentence embedding
      indexer.py          # FAISS index management
    models/
      schemas.py          # Pydantic request/response models
    data/
      transcripts/        # Stored transcript JSON files
      indexes/            # Stored FAISS index files
    utils/
      file_utils.py       # File validation and path helpers
      security.py         # Input sanitization helpers
  frontend/
    index.html            # Single-page UI
  .env.example
  .gitignore
  requirements.txt
```
# Smart-Lecture-Search-Engine
