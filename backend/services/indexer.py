import json
import os

import faiss
import numpy as np

from backend.services.embedder import Embedder

_SCORE_THRESHOLD = 0.25


class LectureIndex:
    def __init__(self, lecture_id: str, data_dir: str) -> None:
        """Configure paths for the FAISS index and chunk metadata files.

        Args:
            lecture_id: UUID string identifying the lecture.
            data_dir: Directory under which index files are stored (data/indexes/).
        """
        self._lecture_id = lecture_id
        self._index_path = os.path.join(data_dir, f"{lecture_id}.faiss")
        self._meta_path = os.path.join(data_dir, f"{lecture_id}_meta.json")
        self._index: faiss.Index | None = None
        self._chunks: list[dict] = []

    def build(self, chunks: list[dict], embedder: Embedder) -> None:
        """Build a FAISS IndexFlatIP index from transcript chunks and persist it.

        Inner product on L2-normalised vectors is equivalent to cosine similarity,
        so scores returned by FAISS are directly in the [0, 1] range.

        Args:
            chunks: List of {"start": float, "end": float, "text": str} dicts.
            embedder: Initialised Embedder instance used to produce vectors.

        Raises:
            ValueError: If chunks is empty.
        """
        if not chunks:
            raise ValueError("Cannot build an index from an empty chunk list.")

        texts = [c["text"] for c in chunks]
        vectors: np.ndarray = embedder.embed_texts(texts)  # (n, 384) float32

        dim = vectors.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)

        os.makedirs(os.path.dirname(self._index_path), exist_ok=True)
        faiss.write_index(index, self._index_path)

        with open(self._meta_path, "w", encoding="utf-8") as fh:
            json.dump(chunks, fh, ensure_ascii=False, indent=2)

        self._index = index
        self._chunks = chunks

    def load(self) -> None:
        """Load an existing FAISS index and its chunk metadata from disk.

        Raises:
            FileNotFoundError: If either the index or metadata file is missing.
        """
        if not os.path.exists(self._index_path):
            raise FileNotFoundError(f"FAISS index not found: {self._index_path}")
        if not os.path.exists(self._meta_path):
            raise FileNotFoundError(f"Metadata file not found: {self._meta_path}")

        self._index = faiss.read_index(self._index_path)

        with open(self._meta_path, "r", encoding="utf-8") as fh:
            self._chunks = json.load(fh)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """Search the index for the closest chunks to a query vector.

        Args:
            query_vector: Float32 array of shape (1, 384), L2-normalised.
            top_k: Maximum number of results to return before score filtering.

        Returns:
            List of matching chunks, each with an added "score" key (float, 0–1),
            ordered by descending score. Results with score < 0.25 are excluded.

        Raises:
            RuntimeError: If the index has not been built or loaded yet.
        """
        if self._index is None:
            raise RuntimeError(
                f"Index for lecture '{self._lecture_id}' is not loaded. "
                "Call build() or load() first."
            )

        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < _SCORE_THRESHOLD:
                continue
            chunk = dict(self._chunks[idx])
            chunk["score"] = round(float(score), 4)
            results.append(chunk)

        return results
