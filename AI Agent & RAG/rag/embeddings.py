from typing import List, Any
import numpy as np
from sentence_transformers import SentenceTransformer


class Embeddings:
    """Embedding generator using sentence-transformers"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the embedding model"""
        try:
            self.model = SentenceTransformer(self.model_name)
        except Exception as e:
            print(f"Error loading embedding model: {str(e)}")
            self.model = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        if not self.model:
            raise ValueError("Embedding model not loaded")

        try:
            embeddings = self.model.encode(
                texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embeddings.tolist()
        except Exception as e:
            raise ValueError(f"Error generating embeddings: {str(e)}")

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a single query"""
        if not self.model:
            raise ValueError("Embedding model not loaded")

        try:
            embedding = self.model.encode(
                query,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            return embedding.tolist()
        except Exception as e:
            raise ValueError(f"Error generating query embedding: {str(e)}")

    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            v1 = np.array(embedding1)
            v2 = np.array(embedding2)
            return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        except Exception:
            return 0.0

    def compute_similarities(self, query_embedding: List[float], doc_embeddings: List[List[float]]) -> List[float]:
        """Compute similarities between query and multiple documents"""
        similarities = []
        for doc_emb in doc_embeddings:
            sim = self.similarity(query_embedding, doc_emb)
            similarities.append(sim)
        return similarities

    def batch_embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """Generate embeddings in batches"""
        if not self.model:
            raise ValueError("Embedding model not loaded")

        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                embeddings = self.model.encode(
                    batch,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False
                )
                all_embeddings.extend(embeddings.tolist())
            except Exception as e:
                raise ValueError(f"Error in batch {i//batch_size}: {str(e)}")

        return all_embeddings
