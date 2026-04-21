from typing import List, Dict, Any
from .embeddings import Embeddings


class Reranker:
    """Reranker for improving retrieval results"""

    def __init__(self, embeddings: Embeddings = None):
        self.embeddings = embeddings

    def rerank(self, query: str, documents: List[str], top_k: int = None) -> List[Dict[str, Any]]:
        """Rerank documents based on relevance to query"""
        if not documents:
            return []

        try:
            if self.embeddings:
                # Use embedding similarity for reranking
                query_embedding = self.embeddings.embed_query(query)
                doc_embeddings = self.embeddings.embed_documents(documents)

                # Compute similarities
                similarities = self.embeddings.compute_similarities(query_embedding, doc_embeddings)

                # Create result list
                results = []
                for i, (doc, score) in enumerate(zip(documents, similarities)):
                    results.append({
                        'document': doc,
                        'score': score,
                        'original_rank': i
                    })

                # Sort by score
                results.sort(key=lambda x: x['score'], reverse=True)
            else:
                # Simple keyword-based reranking
                query_lower = query.lower()
                query_terms = set(query_lower.split())

                results = []
                for i, doc in enumerate(documents):
                    doc_lower = doc.lower()
                    # Count matching terms
                    matches = sum(1 for term in query_terms if term in doc_lower)
                    score = matches / len(query_terms) if query_terms else 0

                    results.append({
                        'document': doc,
                        'score': score,
                        'original_rank': i
                    })

                results.sort(key=lambda x: x['score'], reverse=True)

            # Return top_k if specified
            if top_k:
                return results[:top_k]
            return results

        except Exception as e:
            print(f"Reranking error: {str(e)}")
            # Return original order if reranking fails
            return [{'document': doc, 'score': 0.0, 'original_rank': i} for i, doc in enumerate(documents)]

    def rerank_with_metadata(self, query: str, items: List[Dict[str, Any]], text_key: str = 'document') -> List[Dict[str, Any]]:
        """Rerank items that have metadata"""
        documents = [item[text_key] for item in items]
        reranked = self.rerank(query, documents)

        # Merge scores back into original items
        score_map = {result['document']: result['score'] for result in reranked}

        for item in items:
            item['rerank_score'] = score_map.get(item[text_key], 0.0)

        # Sort by rerank score
        items.sort(key=lambda x: x['rerank_score'], reverse=True)
        return items

    def cross_encoder_rerank(self, query: str, documents: List[str], top_k: int = 10) -> List[Dict[str, Any]]:
        """Advanced reranking using cross-encoder (if available)"""
        try:
            from sentence_transformers import CrossEncoder

            # Load cross-encoder model
            model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

            # Create pairs
            pairs = [(query, doc) for doc in documents]

            # Get scores
            scores = model.predict(pairs)

            # Create results
            results = []
            for i, (doc, score) in enumerate(zip(documents, scores)):
                results.append({
                    'document': doc,
                    'score': float(score),
                    'original_rank': i
                })

            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]

        except ImportError:
            print("Cross-encoder not available, using basic reranking")
            return self.rerank(query, documents, top_k)
        except Exception as e:
            print(f"Cross-encoder error: {str(e)}")
            return self.rerank(query, documents, top_k)

    def filter_by_threshold(self, items: List[Dict[str, Any]], threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Filter items by score threshold"""
        return [item for item in items if item.get('score', 0) >= threshold]

    def diversify_results(self, items: List[Dict[str, Any]], diversity_factor: float = 0.5, top_k: int = 10) -> List[Dict[str, Any]]:
        """Diversify results to reduce redundancy"""
        if len(items) <= 1:
            return items[:top_k]

        diverse_results = []
        remaining = items.copy()

        # Add highest scoring item
        diverse_results.append(remaining.pop(0))

        # Iteratively add items that are least similar to already selected
        while remaining and len(diverse_results) < top_k:
            best_item = None
            best_score = -1

            for item in remaining:
                # Calculate max similarity to existing results
                if self.embeddings:
                    item_embedding = self.embeddings.embed_query(item['document'])
                    max_sim = 0

                    for selected in diverse_results:
                        selected_embedding = self.embeddings.embed_query(selected['document'])
                        sim = self.embeddings.similarity(item_embedding, selected_embedding)
                        max_sim = max(max_sim, sim)

                    # Original score adjusted by diversity
                    adjusted_score = item['score'] * (1 - diversity_factor * max_sim)
                else:
                    adjusted_score = item['score']

                if adjusted_score > best_score:
                    best_score = adjusted_score
                    best_item = item

            if best_item:
                diverse_results.append(best_item)
                remaining.remove(best_item)

        return diverse_results
