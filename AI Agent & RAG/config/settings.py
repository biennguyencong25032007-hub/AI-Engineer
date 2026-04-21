import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    """Application settings configuration"""

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
    WEAVIATE_URL = os.getenv("WEAVIATE_URL", "")
    WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY", "")

    # Model configurations
    DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "gpt-4")
    DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # RAG configurations
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
    TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))

    # Vector store
    VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "chroma")  # chroma, faiss, pinecone, weaviate
    VECTOR_STORE_PATH = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")

    # Memory
    MEMORY_TYPE = os.getenv("MEMORY_TYPE", "short_term")  # short_term, long_term, episodic
    MAX_MEMORY_ITEMS = int(os.getenv("MAX_MEMORY_ITEMS", "10"))

    # Agent
    AGENT_TYPE = os.getenv("AGENT_TYPE", "tool_calling")  # base, react, tool_calling, multi

    # Paths
    DATA_DIR = os.getenv("DATA_DIR", "./data")
    DOCUMENTS_DIR = os.getenv("DOCUMENTS_DIR", "./data/documents")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

    @classmethod
    def validate(cls):
        """Validate required settings"""
        required_vars = []
        if not cls.OPENAI_API_KEY and not cls.ANTHROPIC_API_KEY:
            required_vars.append("OPENAI_API_KEY or ANTHROPIC_API_KEY")

        if required_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(required_vars)}")

        return True
