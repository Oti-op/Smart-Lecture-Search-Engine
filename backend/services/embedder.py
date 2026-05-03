import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBEDDING_DIM = 384


class Embedder:
    def __init__(self, model_name: str = _MODEL_NAME) -> None:
        """Load the sentence-transformers model for dense embedding."""
        self._model = SentenceTransformer(model_name)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        """Embed a list of text chunks and return L2-normalised float32 vectors.

        Args:
            texts: Non-empty list of strings to embed.

        Returns:
            Float32 numpy array of shape (len(texts), 384), L2-normalised so that
            inner product equals cosine similarity.

        Raises:
            ValueError: If texts is empty.
        """
        if not texts:
            raise ValueError("embed_texts received an empty list.")

        vectors = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.astype(np.float32)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string and return a normalised float32 vector.

        Args:
            query: The search query.

        Returns:
            Float32 numpy array of shape (1, 384), L2-normalised.
        """
        vector = self._model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vector.astype(np.float32)
