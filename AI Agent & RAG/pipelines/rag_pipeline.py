"""RAG Pipeline - Complete Retrieval-Augmented Generation pipeline"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from rag.document_loader import DocumentLoader
from rag.text_splitter import TextSplitter
from rag.embeddings import Embeddings
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.reranker import Reranker
from config.settings import Settings
from config.prompts import RAG_PROMPT_TEMPLATE


class RAGPipeline:
    """Complete RAG pipeline from documents to answers"""

    def __init__(self, config: Dict[str, Any] = None):
        self.settings = Settings()
        self.config = config or {}

        # Initialize components
        self.document_loader = DocumentLoader()
        self.text_splitter = TextSplitter()
        self.embeddings = Embeddings(
            model_name=self.config.get('embedding_model', self.settings.DEFAULT_EMBEDDING_MODEL)
        )
        self.vector_store = VectorStore(
            store_path=self.config.get('vector_store_path', self.settings.VECTOR_STORE_PATH)
        )
        self.retriever = Retriever(self.vector_store, self.embeddings)
        self.reranker = Reranker(self.embeddings)

        self.is_indexed = False

    def load_documents(self, documents_dir: str = None) -> List[Any]:
        """Load documents from directory"""
        docs_dir = documents_dir or self.settings.DOCUMENTS_DIR

        if not Path(docs_dir).exists():
            raise FileNotFoundError(f"Documents directory not found: {docs_dir}")

        documents = []
        for filepath in Path(docs_dir).glob("**/*"):
            if filepath.suffix in ['.txt', '.md', '.pdf', '.docx']:
                try:
                    docs = self.document_loader.load(str(filepath))
                    documents.extend(docs)
                except Exception as e:
                    print(f"Error loading {filepath}: {str(e)}")

        return documents

    def split_documents(self, documents: List[Any]) -> List[Any]:
        """Split documents into chunks"""
        return self.text_splitter.split(documents)

    def index_documents(self, documents_dir: str = None, rebuild: bool = False):
        """Complete indexing pipeline"""
        if self.is_indexed and not rebuild:
            print("Documents already indexed. Use rebuild=True to reindex.")
            return

        print("Loading documents...")
        documents = self.load_documents(documents_dir)
        print(f"Loaded {len(documents)} documents")

        print("Splitting documents...")
        chunks = self.split_documents(documents)
        print(f"Created {len(chunks)} chunks")

        print("Generating embeddings...")
        texts = [chunk.page_content for chunk in chunks]
        embeddings = self.embeddings.embed_documents(texts)

        print("Storing in vector store...")
        metadata = [{'source': getattr(chunk, 'metadata', {})} for chunk in chunks]
        self.vector_store.add_documents(texts, embeddings, metadata)

        self.is_indexed = True
        print("Indexing complete!")

    def retrieve(self, query: str, k: int = None, use_reranker: bool = True) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for query"""
        k = k or self.settings.TOP_K_RETRIEVAL

        # Initial retrieval
        results = self.retriever.retrieve(query, k=k * 2)  # Get more for reranking

        if use_reranker and results:
            # Rerank results
            documents = [result['document'] for result in results]
            reranked = self.reranker.rerank(query, documents, top_k=k)

            # Map back to original results
            doc_to_result = {result['document']: result for result in results}
            final_results = []
            for item in reranked:
                if item['document'] in doc_to_result:
                    result = doc_to_result[item['document']].copy()
                    result['rerank_score'] = item['score']
                    final_results.append(result)

            return final_results[:k]

        return results[:k]

    def generate(self, query: str, k: int = None) -> str:
        """Generate answer for query using RAG"""
        try:
            # Retrieve relevant documents
            results = self.retrieve(query, k=k)

            if not results:
                return "I couldn't find relevant information to answer your question."

            # Build context
            context_parts = []
            for i, result in enumerate(results):
                context_parts.append(f"[{i+1}] {result['document']}")

            context = "\n\n".join(context_parts)

            # Generate answer using LLM
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=query
            )

            answer = self._generate_answer(prompt)

            return answer
        except Exception as e:
            return f"RAG generation error: {str(e)}"

    def _generate_answer(self, prompt: str) -> str:
        """Generate answer using configured LLM"""
        if self.settings.OPENAI_API_KEY:
            return self._call_openai(prompt)
        elif self.settings.ANTHROPIC_API_KEY:
            return self._call_anthropic(prompt)
        else:
            return "No LLM API key configured. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY."

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        from openai import OpenAI
        client = OpenAI(api_key=self.settings.OPENAI_API_KEY)

        try:
            response = client.chat.completions.create(
                model=self.settings.DEFAULT_LLM_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API error: {str(e)}"

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        from anthropic import Anthropic
        client = Anthropic(api_key=self.settings.ANTHROPIC_API_KEY)

        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return f"Anthropic API error: {str(e)}"

    def query(self, question: str, k: int = None) -> Dict[str, Any]:
        """Full RAG query - retrieve and generate"""
        results = self.retrieve(question, k=k)
        answer = self.generate(question, k=k)

        return {
            'question': question,
            'answer': answer,
            'sources': [
                {
                    'content': r['document'],
                    'score': r.get('rerank_score', r.get('similarity', 0)),
                    'metadata': r.get('metadata', {})
                }
                for r in results
            ]
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'is_indexed': self.is_indexed,
            'document_count': self.vector_store.count(),
            'embedding_model': self.embeddings.model_name
        }

    def clear_index(self):
        """Clear all indexed documents"""
        self.vector_store.clear()
        self.is_indexed = False
