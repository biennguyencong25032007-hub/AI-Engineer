"""
RAG Pipeline - Retrieval Augmented Generation
Complete RAG implementation with chunking, embedding, và retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.config import config, RAGConfig
from src.logger import get_logger
from src.llm.base import BaseLLMClient, Message
from src.rag.chunking import TextSplitter, Chunk
from src.rag.embedding import get_embedder, BaseEmbedder
from src.rag.vector_db import VectorDBFactory, BaseVectorDB, SearchResult

logger = get_logger("rag")


@dataclass
class RetrievedContext:
    """Retrieved context with relevance information."""
    chunks: list[SearchResult]
    query: str
    total_tokens: int = 0

    def to_prompt(self, max_length: int = 4000) -> str:
        """Format context as prompt string."""
        context_parts = []
        total_len = 0

        for chunk in self.chunks:
            part = f"[Source {chunk.metadata.get('source', 'unknown')}]\n{chunk.text}\n"
            if total_len + len(part) > max_length:
                break
            context_parts.append(part)
            total_len += len(part)

        return "\n".join(context_parts)


class RAGPipeline:
    """
    Complete RAG pipeline:

    1. Index documents (chunk → embed → store)
    2. Query (embed query → retrieve → rerank)
    3. Generate (build prompt with context)
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        vector_db: BaseVectorDB | None = None,
        embedder: BaseEmbedder | None = None,
        chunker: TextSplitter | None = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ):
        self.llm = llm
        self.vector_db = vector_db or VectorDBFactory.create()
        self.embedder = embedder or get_embedder()
        self.chunker = chunker or TextSplitter(
            chunk_size=config.rag.chunk_size,
            chunk_overlap=config.rag.chunk_overlap,
        )
        self.top_k = top_k
        self.min_similarity = min_similarity

        logger.info(f"RAG Pipeline initialized | top_k={top_k}")

    # ════════════════════════════════════════════════════════════════════
    # Indexing
    # ════════════════════════════════════════════════════════════════════
    def index_text(
        self,
        text: str,
        doc_id: str,
        metadata: dict | None = None,
    ) -> int:
        """
        Index a text document.

        Returns:
            Number of chunks indexed
        """
        metadata = metadata or {}

        # Chunk the text
        chunks = self.chunker.split_text(text, {**metadata, "doc_id": doc_id})

        if not chunks:
            logger.warning(f"No chunks generated for doc_id={doc_id}")
            return 0

        # Generate embeddings
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed(texts)

        # Store in vector DB
        ids = [f"{doc_id}_{c.index}" for c in chunks]
        metadatas = [
            {**c.metadata, "chunk_index": c.index}
            for c in chunks
        ]

        self.vector_db.add(ids, embeddings, texts, metadatas)

        logger.info(f"Indexed {len(chunks)} chunks for doc_id={doc_id}")
        return len(chunks)

    def index_file(self, file_path: str | Path, metadata: dict | None = None) -> int:
        """Index a text file."""
        path = Path(file_path)
        metadata = metadata or {}

        # Read file
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        # Add source metadata
        metadata["source"] = str(path)
        metadata["filename"] = path.name

        return self.index_text(text, doc_id=path.stem, metadata=metadata)

    def index_directory(
        self,
        dir_path: str | Path,
        extensions: list[str] = [".txt", ".md", ".pdf"],
        recursive: bool = True,
    ) -> int:
        """Index all files in a directory."""
        dir_path = Path(dir_path)
        total_chunks = 0

        patterns = ["**/*" + ext for ext in extensions] if recursive else ["*" + ext for ext in extensions]

        for pattern in patterns:
            for file_path in dir_path.glob(pattern):
                if file_path.is_file():
                    try:
                        chunks = self.index_file(file_path)
                        total_chunks += chunks
                    except Exception as e:
                        logger.error(f"Failed to index {file_path}: {e}")

        logger.info(f"Indexed {total_chunks} chunks from {dir_path}")
        return total_chunks

    # ════════════════════════════════════════════════════════════════════
    # Retrieval
    # ════════════════════════════════════════════════════════════════════
    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> RetrievedContext:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            filter_metadata: Optional metadata filters

        Returns:
            RetrievedContext with relevant chunks
        """
        top_k = top_k or self.top_k

        # Embed query
        query_embedding = self.embedder.embed_single(query)

        # Search vector DB
        results = self.vector_db.search(
            query_embedding=query_embedding,
            top_k=top_k,
            filter_metadata=filter_metadata,
        )

        # Filter by minimum similarity
        results = [r for r in results if r.score >= self.min_similarity]

        # Estimate token count
        total_tokens = sum(len(r.text) // 4 for r in results)

        return RetrievedContext(
            chunks=results,
            query=query,
            total_tokens=total_tokens,
        )

    # ════════════════════════════════════════════════════════════════════
    # Generation
    # ════════════════════════════════════════════════════════════════════
    def generate_with_context(
        self,
        query: str,
        system_prompt: str = "You are a helpful assistant.",
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> tuple[str, RetrievedContext]:
        """
        Generate answer with retrieved context.

        Returns:
            (generated_answer, retrieved_context)
        """
        # Retrieve context
        context = self.retrieve(query, top_k, filter_metadata)

        if not context.chunks:
            logger.warning("No relevant context found")

        # Build prompt
        context_text = context.to_prompt()

        if context_text:
            user_prompt = f"""Answer the question based on the following context.

Context:
{context_text}

Question: {query}

Answer:"""
        else:
            user_prompt = query

        # Generate response
        response = self.llm.chat(
            messages=[Message(role="user", content=user_prompt)],
            system=system_prompt,
        )

        return response.content, context

    def query(self, question: str) -> dict:
        """
        Full RAG query with structured output.

        Returns:
            Dictionary with answer, sources, và metadata
        """
        answer, context = self.generate_with_context(question)

        return {
            "answer": answer,
            "sources": [
                {
                    "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                    "score": r.score,
                    "metadata": r.metadata,
                }
                for r in context.chunks
            ],
            "query": context.query,
            "num_sources": len(context.chunks),
        }

    # ════════════════════════════════════════════════════════════════════
    # Utilities
    # ════════════════════════════════════════════════════════════════════
    def get_stats(self) -> dict:
        """Get pipeline statistics."""
        return {
            "total_documents": self.vector_db.count(),
            "embedder": self.embedder.__class__.__name__,
            "vector_db": self.vector_db.__class__.__name__,
            "chunk_size": self.chunker.chunk_size,
        }
