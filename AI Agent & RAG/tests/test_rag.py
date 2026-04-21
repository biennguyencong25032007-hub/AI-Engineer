"""Tests for RAG modules"""

import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.embeddings import Embeddings
from rag.vector_store import VectorStore
from rag.text_splitter import TextSplitter
from rag.document_loader import DocumentLoader


def test_embeddings():
    """Test Embeddings module"""
    print("\nTesting Embeddings...")
    
    emb = Embeddings()
    
    # Test single embedding
    text = "Hello world"
    embedding = emb.embed_query(text)
    assert isinstance(embedding, list), "Embedding should be a list"
    assert len(embedding) > 0, "Embedding should not be empty"
    
    # Test batch embeddings
    texts = ["Hello", "World", "Test"]
    embeddings = emb.embed_documents(texts)
    assert len(embeddings) == len(texts), "Should get same number of embeddings as texts"
    assert all(isinstance(e, list) for e in embeddings), "All embeddings should be lists"
    
    # Test similarity
    sim = emb.similarity(embedding, embedding)
    assert abs(sim - 1.0) < 0.01, "Self-similarity should be ~1.0"
    
    print("✓ Embeddings works correctly")
    return True


def test_vector_store():
    """Test VectorStore"""
    print("\nTesting VectorStore...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(store_path=tmpdir)
        
        # Test adding documents
        docs = ["Doc 1 content", "Doc 2 content", "Doc 3 content"]
        embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]
        metadata = [{'id': 1}, {'id': 2}, {'id': 3}]
        
        store.add_documents(docs, embeddings, metadata)
        assert store.count() == 3, f"Expected 3 docs, got {store.count()}"
        
        # Test retrieval
        query_embedding = [0.15, 0.25, 0.35]
        results = store.similarity_search(query_embedding, k=2)
        
        assert len(results) <= 2, "Should return at most k results"
        assert all('document' in r for r in results), "Results should contain documents"
        assert all('similarity' in r for r in results), "Results should contain scores"
        
        # Test delete
        deleted = store.delete(0)
        assert deleted, "Delete should return True"
        assert store.count() == 2, f"Expected 2 docs after delete, got {store.count()}"
        
        # Test clear
        store.clear()
        assert store.count() == 0, "Clear should remove all docs"
    
    print("✓ VectorStore works correctly")
    return True


def test_text_splitter():
    """Test TextSplitter"""
    print("\nTesting TextSplitter...")
    
    splitter = TextSplitter()
    
    # Create mock documents
    class MockDoc:
        def __init__(self, content):
            self.page_content = content
            self.metadata = {}
    
    docs = [MockDoc("This is a test document. " * 50)]
    
    splits = splitter.split(docs)
    
    assert len(splits) > 1, "Long document should be split into multiple chunks"
    assert all(hasattr(s, 'page_content') for s in splits), "Splits should have page_content"
    
    print(f"✓ TextSplitter works correctly (created {len(splits)} chunks)")
    return True


def test_document_loader():
    """Test DocumentLoader"""
    print("\nTesting DocumentLoader...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test file
        test_file = Path(tmpdir) / "test.txt"
        test_file.write_text("This is a test document.\nSecond line.")
        
        loader = DocumentLoader()
        docs = loader.load(str(test_file))
        
        assert len(docs) > 0, "Should load at least one document"
        assert hasattr(docs[0], 'page_content'), "Document should have page_content"
        assert "test document" in docs[0].page_content.lower(), "Document content mismatch"
    
    print("✓ DocumentLoader works correctly")
    return True


def run_all_tests():
    """Run all RAG tests"""
    print("=" * 50)
    print("RUNNING RAG TESTS")
    print("=" * 50)
    
    tests = [
        test_embeddings,
        test_vector_store,
        test_text_splitter,
        test_document_loader
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
