"""
Vector Database Implementations
Support for ChromaDB, FAISS, và in-memory storage.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

# Optional logger
try:
    from src.config import config
    from src.logger import get_logger
    logger = get_logger("vector_db")
except ImportError:
    import logging
    logger = logging.getLogger("vector_db")
    class config:
        class vector_db:
            persist_directory = Path("./data/chroma_db")
            distance_metric = "cosine"


@dataclass
class SearchResult:
    """A search result with score and metadata."""
    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


class BaseVectorDB(ABC):
    """Abstract base for vector databases."""

    @abstractmethod
    def add(self, ids: list[str], embeddings: np.ndarray, texts: list[str], metadata: list[dict]) -> None:
        """Add documents to the database."""
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        pass

    @abstractmethod
    def delete(self, ids: list[str]) -> None:
        """Delete documents by ID."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Count total documents."""
        pass


class ChromaDB(BaseVectorDB):
    """
    ChromaDB vector database.

    Persistent storage với metadata filtering.
    """

    def __init__(
        self,
        persist_directory: Path | str | None = None,
        collection_name: str = "documents",
    ):
        import chromadb
        from chromadb.config import Settings

        if persist_directory is None:
            persist_directory = config.vector_db.persist_directory

        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": config.vector_db.distance_metric},
        )

        logger.info(f"ChromaDB initialized: {collection_name}")

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        texts: list[str],
        metadata: list[dict],
    ) -> None:
        """Add documents."""
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadata,
        )
        logger.debug(f"Added {len(ids)} documents to ChromaDB")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        if results["ids"]:
            for i, doc_id in enumerate(results["ids"][0]):
                search_results.append(SearchResult(
                    id=doc_id,
                    text=results["documents"][0][i],
                    score=1 - results["distances"][0][i],  # Convert distance to similarity
                    metadata=results["metadatas"][0][i] or {},
                ))

        return search_results

    def delete(self, ids: list[str]) -> None:
        """Delete documents."""
        self.collection.delete(ids=ids)

    def count(self) -> int:
        """Count documents."""
        return self.collection.count()


class FAISSVectorDB(BaseVectorDB):
    """
    FAISS vector database.

    Fast in-memory search, good cho large datasets.
    """

    def __init__(self, dimension: int = 384, metric: str = "cosine"):
        import faiss

        self.dimension = dimension
        self.metric = metric

        # Create index based on metric
        if metric == "cosine":
            # Normalized vectors, use inner product
            self.index = faiss.IndexFlatIP(dimension)
        elif metric == "euclidean":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            self.index = faiss.IndexFlatIP(dimension)

        # In-memory storage
        self._ids: list[str] = []
        self._texts: list[str] = []
        self._metadata: list[dict] = []

        logger.info(f"FAISS initialized: dim={dimension}, metric={metric}")

    def add(
        self,
        ids: list[str],
        embeddings: np.ndarray,
        texts: list[str],
        metadata: list[dict],
    ) -> None:
        """Add documents."""
        # Normalize for cosine similarity
        if self.metric == "cosine":
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        self.index.add(embeddings.astype(np.float32))
        self._ids.extend(ids)
        self._texts.extend(texts)
        self._metadata.extend(metadata)

        logger.debug(f"Added {len(ids)} documents to FAISS")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        """Search for similar documents."""
        if self.index.ntotal == 0:
            return []

        # Normalize query
        if self.metric == "cosine":
            query_embedding = query_embedding / np.linalg.norm(query_embedding)

        # Search
        scores, indices = self.index.search(
            query_embedding.reshape(1, -1).astype(np.float32),
            min(top_k, self.index.ntotal),
        )

        results = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue

            # Apply metadata filter
            if filter_metadata:
                doc_meta = self._metadata[idx]
                if not all(doc_meta.get(k) == v for k, v in filter_metadata.items()):
                    continue

            results.append(SearchResult(
                id=self._ids[idx],
                text=self._texts[idx],
                score=float(score),
                metadata=self._metadata[idx],
            ))

        return results

    def delete(self, ids: list[str]) -> None:
        """Delete documents (marks as removed, rebuild for true deletion)."""
        # FAISS doesn't support deletion, flag for rebuilding
        removed_indices = [i for i, id_ in enumerate(self._ids) if id_ in ids]
        for idx in sorted(removed_indices, reverse=True):
            self._ids.pop(idx)
            self._texts.pop(idx)
            self._metadata.pop(idx)

        # Rebuild index
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild FAISS index after deletions."""
        import faiss

        self.index.reset()
        if self._texts:
            embeddings = np.array([
                self._metadata[i].get("embedding")
                for i in range(len(self._texts))
            ], dtype=np.float32)

            if len(embeddings.shape) == 1:
                return

            if self.metric == "cosine":
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

            self.index.add(embeddings)

    def count(self) -> int:
        """Count documents."""
        return self.index.ntotal


class InMemoryDB(BaseVectorDB):
    """Simple in-memory vector store for small datasets."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._data: list[dict] = []
        logger.info("In-memory DB initialized")

    def add(self, ids: list[str], embeddings: np.ndarray, texts: list[str], metadata: list[dict]) -> None:
        for i, id_ in enumerate(ids):
            self._data.append({
                "id": id_,
                "embedding": embeddings[i],
                "text": texts[i],
                "metadata": metadata[i],
            })

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_metadata: dict | None = None,
    ) -> list[SearchResult]:
        if not self._data:
            return []

        # Compute similarities
        results = []
        for item in self._data:
            if filter_metadata:
                if not all(item["metadata"].get(k) == v for k, v in filter_metadata.items()):
                    continue

            score = float(np.dot(query_embedding, item["embedding"]))
            results.append(SearchResult(
                id=item["id"],
                text=item["text"],
                score=score,
                metadata=item["metadata"],
            ))

        # Sort and limit
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    def delete(self, ids: list[str]) -> None:
        self._data = [d for d in self._data if d["id"] not in ids]

    def count(self) -> int:
        return len(self._data)


class VectorDBFactory:
    """Factory for vector database instances."""

    @classmethod
    def create(cls, provider: str | None = None, **kwargs) -> BaseVectorDB:
        """Create a vector DB by provider."""
        if provider is None:
            provider = config.vector_db.provider

        if provider == "chroma":
            return ChromaDB(**kwargs)
        elif provider == "faiss":
            return FAISSVectorDB(**kwargs)
        elif provider == "memory":
            return InMemoryDB(**kwargs)
        else:
            raise ValueError(f"Unknown vector DB provider: {provider}")
