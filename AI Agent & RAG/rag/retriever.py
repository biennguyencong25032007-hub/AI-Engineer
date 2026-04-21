from typing import List, Dict, Any
from .vector_store import VectorStore
from .embeddings import Embeddings


class Retriever:
    """Retriever for fetching relevant documents based on query"""

    def __init__(self, vector_store: VectorStore, embeddings: Embeddings):
        self.vector_store = vector_store
        self.embeddings = embeddings

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top k most relevant documents for query"""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)

            # Search in vector store
            results = self.vector_store.similarity_search(query_embedding, k=k)

            return results
        except Exception as e:
            print(f"Retrieval error: {str(e)}")
            return []

    def retrieve_with_scores(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents with similarity scores"""
        results = self.retrieve(query, k)
        # Results already include similarity scores from vector_store
        return results

    def retrieve_hybrid(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
        """Hybrid retrieval combining vector and keyword search"""
        vector_results = self.retrieve(query, k=k)

        # Simple keyword boosting (could be enhanced)
        keyword_boosted = []
        for result in vector_results:
            doc = result['document'].lower()
            query_terms = query.lower().split()

            # Count matching terms
            matches = sum(1 for term in query_terms if term in doc)
            keyword_score = matches / len(query_terms) if query_terms else 0

            # Combine scores
            combined_score = (alpha * result['similarity']) + ((1 - alpha) * keyword_score)
            result['combined_score'] = combined_score
            keyword_boosted.append(result)

        # Sort by combined score
        keyword_boosted.sort(key=lambda x: x['combined_score'], reverse=True)
        return keyword_boosted[:k]

    def batch_retrieve(self, queries: List[str], k: int = 5) -> List[List[Dict[str, Any]]]:
        """Retrieve for multiple queries"""
        all_results = []
        for query in queries:
            results = self.retrieve(query, k=k)
            all_results.append(results)
        return all_results

    def retrieve_with_metadata_filter(self, query: str, filters: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve with metadata filtering"""
        results = self.retrieve(query, k=k * 2)  # Get more to filter

        filtered = []
        for result in results:
            metadata = result.get('metadata', {})
            match = True

            for key, value in filters.items():
                if key not in metadata or metadata[key] != value:
                    match = False
                    break

            if match:
                filtered.append(result)
                if len(filtered) >= k:
                    break

        return filtered

    def get_relevant_documents_text(self, query: str, k: int = 5) -> str:
        """Get relevant documents formatted as text"""
        results = self.retrieve(query, k=k)

        if not results:
            return "No relevant documents found."

        formatted = []
        for i, result in enumerate(results, 1):
            doc = result['document']
            score = result.get('similarity', 0)
            metadata = result.get('metadata', {})

            text = f"[Document {i}]\n"
            if metadata:
                meta_str = ", ".join(f"{k}: {v}" for k, v in metadata.items())
                text += f"Metadata: {meta_str}\n"
            text += f"Relevance: {score:.3f}\n"
            text += f"Content: {doc}\n"
            formatted.append(text)

        return "\n".join(formatted)
