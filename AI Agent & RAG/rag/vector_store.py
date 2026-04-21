import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional


class VectorStore:
    """Simple vector store for storing and retrieving embeddings"""

    def __init__(self, store_path: str = "./data/vector_store"):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)

        self.documents = []
        self.embeddings = []
        self.metadata = []

        self._index_file = self.store_path / "index.pkl"
        self._documents_file = self.store_path / "documents.pkl"

        self._load()

    def add_documents(self, documents: List[str], embeddings: List[List[float]], metadata: List[Dict[str, Any]] = None):
        """Add documents and their embeddings to the store"""
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match")

        for i, doc in enumerate(documents):
            self.documents.append(doc)
            self.embeddings.append(embeddings[i])

            if metadata and i < len(metadata):
                self.metadata.append(metadata[i])
            else:
                self.metadata.append({})

        self._save()

    def add(self, document: str, embedding: List[float], metadata: Dict[str, Any] = None):
        """Add a single document with embedding"""
        self.documents.append(document)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
        self._save()

    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar documents to query embedding"""
        if not self.embeddings:
            return []

        try:
            import numpy as np

            # Convert to numpy arrays
            query_vec = np.array(query_embedding)
            doc_vecs = np.array(self.embeddings)

            # Compute cosine similarities
            norms = np.linalg.norm(doc_vecs, axis=1, keepdims=True)
            normalized = doc_vecs / norms
            similarities = np.dot(normalized, query_vec)

            # Get top k indices
            top_k_indices = np.argsort(similarities)[-k:][::-1]

            results = []
            for idx in top_k_indices:
                results.append({
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'similarity': float(similarities[idx]),
                    'index': int(idx)
                })

            return results
        except Exception as e:
            print(f"Error in similarity search: {str(e)}")
            return []

    def get_document(self, index: int) -> Optional[Dict[str, Any]]:
        """Retrieve document by index"""
        if 0 <= index < len(self.documents):
            return {
                'document': self.documents[index],
                'metadata': self.metadata[index],
                'index': index
            }
        return None

    def delete(self, index: int) -> bool:
        """Delete document by index"""
        if 0 <= index < len(self.documents):
            del self.documents[index]
            del self.embeddings[index]
            del self.metadata[index]
            self._save()
            return True
        return False

    def clear(self):
        """Clear all documents from store"""
        self.documents.clear()
        self.embeddings.clear()
        self.metadata.clear()
        self._save()

    def count(self) -> int:
        """Return number of documents in store"""
        return len(self.documents)

    def _save(self):
        """Save store to disk"""
        try:
            with open(self._index_file, 'wb') as f:
                pickle.dump({
                    'embeddings': self.embeddings,
                    'metadata': self.metadata
                }, f)

            with open(self._documents_file, 'wb') as f:
                pickle.dump(self.documents, f)
        except Exception as e:
            print(f"Error saving vector store: {str(e)}")

    def _load(self):
        """Load store from disk"""
        try:
            if self._index_file.exists() and self._documents_file.exists():
                with open(self._index_file, 'rb') as f:
                    data = pickle.load(f)
                    self.embeddings = data.get('embeddings', [])
                    self.metadata = data.get('metadata', [])

                with open(self._documents_file, 'rb') as f:
                    self.documents = pickle.load(f)
        except Exception as e:
            print(f"Error loading vector store: {str(e)}")
            self.documents = []
            self.embeddings = []
            self.metadata = []

    def to_dict(self) -> Dict[str, Any]:
        """Export store to dictionary"""
        return {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], store_path: str = None) -> 'VectorStore':
        """Create store from dictionary"""
        store = cls(store_path or "./data/vector_store")
        store.documents = data.get('documents', [])
        store.embeddings = data.get('embeddings', [])
        store.metadata = data.get('metadata', [])
        return store
