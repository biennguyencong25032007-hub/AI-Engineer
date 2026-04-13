"""
GenAI Project Configuration
Quản lý cấu hình từ config.yaml và environment variables.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Load .env file (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False
    BaseModel = object
    Field = lambda *args, **kwargs: lambda x: x


# ════════════════════════════════════════════════════════════════════════
# LLM Providers Config
# ════════════════════════════════════════════════════════════════════════
class LLMConfig(BaseModel):
    provider: Literal["anthropic", "openai", "groq", "ollama", "mock"] = "anthropic"
    model: str = "claude-3-5-sonnet-20241022"
    fallback_model: str | None = None

    # Generation settings
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] = Field(default_factory=list)

    # API settings
    api_base: str | None = None  # For custom endpoints
    timeout: int = 60
    max_retries: int = 3


# ════════════════════════════════════════════════════════════════════════
# Embedding Config
# ════════════════════════════════════════════════════════════════════════
class EmbeddingConfig(BaseModel):
    provider: Literal["openai", "sentence-transformers", "mock"] = "sentence-transformers"
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True


# ════════════════════════════════════════════════════════════════════════
# Vector DB Config
# ════════════════════════════════════════════════════════════════════════
class VectorDBConfig(BaseModel):
    provider: Literal["chroma", "faiss", "mock"] = "chroma"
    persist_directory: Path = Path("./data/chroma_db")
    collection_name: str = "documents"
    distance_metric: Literal["cosine", "euclidean", "manhattan"] = "cosine"


# ════════════════════════════════════════════════════════════════════════
# RAG Config
# ════════════════════════════════════════════════════════════════════════
class RAGConfig(BaseModel):
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    min_similarity: float = 0.0
    rerank: bool = False
    hybrid_search: bool = False  # BM25 + vector


# ════════════════════════════════════════════════════════════════════════
# Agent Config
# ════════════════════════════════════════════════════════════════════════
class AgentConfig(BaseModel):
    system_prompt: str = "You are a helpful AI assistant."
    max_iterations: int = 10
    tool_call_limit: int = 50
    verbose: bool = False
    allow_confirmation: bool = False


# ════════════════════════════════════════════════════════════════════════
# Main Config
# ════════════════════════════════════════════════════════════════════════
@dataclass
class Config:
    """Main configuration class."""

    # API Keys
    anthropic_api_key: str | None = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    groq_api_key: str | None = field(default_factory=lambda: os.getenv("GROQ_API_KEY"))

    # Components
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_db: VectorDBConfig = field(default_factory=VectorDBConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Paths
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", "./cache")))
    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "./logs")))
    data_dir: Path = field(default_factory=lambda: Path("./data"))

    # Debug
    debug: bool = False
    verbose: bool = False

    def setup(self) -> None:
        """Ensure directories exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.vector_db.persist_directory.mkdir(parents=True, exist_ok=True)


# Global config instance
config = Config()
