"""
Embedding Utilities
Generate embeddings using various providers.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Optional

import numpy as np

# Optional logger
try:
    from src.config import config, EmbeddingConfig
    from src.logger import get_logger
    logger = get_logger("embedding")
except ImportError:
    import logging
    logger = logging.getLogger("embedding")
    class config:
        class embedding:
            batch_size = 32
            normalize = True
        class EmbeddingConfig:
            batch_size = 32
            normalize = True


class BaseEmbedder:
    """Abstract base for embedding models."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for texts."""
        pass

    @abstractmethod
    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding dimension."""
        pass


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Embeddings using Sentence Transformers.

    Default: sentence-transformers/all-MiniLM-L6-v2 (384 dim)
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self._model = SentenceTransformer(model)
        logger.info(f"Loaded embedding model: {model}")

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate batch embeddings."""
        if not texts:
            return np.array([])

        embeddings = self._model.encode(
            texts,
            batch_size=config.embedding.batch_size,
            show_progress_bar=len(texts) > 10,
            normalize_embeddings=config.embedding.normalize,
            convert_to_numpy=True,
        )
        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()


class OpenAIEmbedder(BaseEmbedder):
    """Embeddings using OpenAI's API."""

    def __init__(self, model: str = "text-embedding-3-small"):
        import openai

        api_key = config.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")

        self._client = openai.OpenAI(api_key=api_key)
        self.model = model

        # Known dimensions for OpenAI models
        self._dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate batch embeddings via API."""
        if not texts:
            return np.array([])

        response = self._client.embeddings.create(
            model=self.model,
            input=texts,
        )

        embeddings = np.array([item.embedding for item in response.data])

        if config.embedding.normalize:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return embeddings

    def embed_single(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimensions.get(self.model, 1536)


class EmbeddingFactory:
    """Factory for creating embedder instances."""

    _embedders: dict = {}

    @classmethod
    def create(cls, provider: str | None = None) -> BaseEmbedder:
        """Create an embedder by provider."""
        if provider is None:
            provider = config.embedding.provider

        if provider == "sentence-transformers":
            return SentenceTransformerEmbedder(
                model=config.embedding.model
            )
        elif provider == "openai":
            return OpenAIEmbedder(model=config.embedding.model)
        elif provider == "mock":
            return MockEmbedder(dimension=config.embedding.dimension)
        else:
            raise ValueError(f"Unknown embedding provider: {provider}")

    @classmethod
    def get_default(cls) -> BaseEmbedder:
        """Get or create cached default embedder."""
        cache_key = f"{config.embedding.provider}:{config.embedding.model}"

        if cache_key not in cls._embedders:
            cls._embedders[cache_key] = cls.create()

        return cls._embedders[cache_key]


class MockEmbedder(BaseEmbedder):
    """Mock embedder for testing."""

    def __init__(self, dimension: int = 384):
        self._dimension = dimension

    def embed(self, texts: list[str]) -> np.ndarray:
        """Generate random embeddings."""
        embeddings = np.random.randn(len(texts), self._dimension).astype(np.float32)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    def embed_single(self, text: str) -> np.ndarray:
        return self.embed([text])[0]

    @property
    def dimension(self) -> int:
        return self._dimension


def get_embedder(provider: str | None = None) -> BaseEmbedder:
    """Quick access to embedder factory."""
    return EmbeddingFactory.create(provider)
