"""
RAG Tests - Standalone test file với direct imports
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Install deps if needed
def ensure_deps():
    import importlib
    deps = ['numpy']
    for dep in deps:
        try:
            importlib.import_module(dep)
        except ImportError:
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", dep, "-q"])

ensure_deps()

import numpy as np

# Direct module loading to avoid __init__.py chain
import importlib.util

def load_source(name, path):
    """Load module directly from file."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

# Load modules
chunking = load_source('chunking', project_root / 'src/rag/chunking.py')
TextSplitter = chunking.TextSplitter
Chunk = chunking.Chunk

vector_db = load_source('vector_db', project_root / 'src/rag/vector_db.py')
InMemoryDB = vector_db.InMemoryDB
SearchResult = vector_db.SearchResult


class MockEmbedder:
    """Mock embedder for testing."""
    def __init__(self, dimension=384):
        self._dimension = dimension

    def embed(self, texts):
        embeddings = np.random.randn(len(texts), self._dimension).astype(np.float32)
        # Normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / (norms + 1e-8)

    def embed_single(self, text):
        return self.embed([text])[0]

    @property
    def dimension(self):
        return self._dimension


# ════════════════════════════════════════════════════════════════════════
# TESTS
# ════════════════════════════════════════════════════════════════════════

class TestTextSplitter:
    """Test text chunking."""

    def test_split_short_text(self):
        splitter = TextSplitter(chunk_size=100)
        chunks = splitter.split_text("Short text.", {})

        assert len(chunks) == 1
        assert "Short text" in chunks[0].text

    def test_split_long_text(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "A" * 200
        chunks = splitter.split_text(text, {})

        # Should split into multiple chunks
        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.text) <= splitter.chunk_size + splitter.chunk_overlap + 20

    def test_empty_text(self):
        splitter = TextSplitter()
        chunks = splitter.split_text("", {})
        assert len(chunks) == 0

    def test_split_with_metadata(self):
        splitter = TextSplitter(chunk_size=50)
        text = "Hello world. This is a test."
        meta = {"source": "test.txt"}
        chunks = splitter.split_text(text, meta)

        assert all("source" in c.metadata for c in chunks)


class TestVectorDB:
    """Test vector database."""

    def test_in_memory_db_add_and_count(self):
        db = InMemoryDB(dimension=384)
        embedder = MockEmbedder(dimension=384)

        embeddings = embedder.embed(["test1", "test2"])
        db.add(["1", "2"], embeddings, ["text1", "text2"], [{"id": 1}, {"id": 2}])

        assert db.count() == 2

    def test_in_memory_db_search(self):
        db = InMemoryDB(dimension=384)
        embedder = MockEmbedder(dimension=384)

        texts = ["apple fruit", "car vehicle", "python programming"]
        embeddings = embedder.embed(texts)

        db.add(["1", "2", "3"], embeddings, texts, [{"type": "fruit"}, {"type": "vehicle"}, {"type": "code"}])

        results = db.search(embeddings[0], top_k=2)

        assert len(results) >= 1
        assert results[0].id == "1"

    def test_in_memory_db_delete(self):
        db = InMemoryDB(dimension=384)
        embedder = MockEmbedder(dimension=384)

        embeddings = embedder.embed(["test1", "test2"])
        db.add(["1", "2"], embeddings, ["text1", "text2"], [{}, {}])

        assert db.count() == 2
        db.delete(["1"])
        assert db.count() == 1


class TestMockEmbedder:
    """Test mock embedder."""

    def test_embed_batch(self):
        embedder = MockEmbedder(dimension=128)
        texts = ["hello", "world"]
        embeddings = embedder.embed(texts)

        assert embeddings.shape == (2, 128)

    def test_embed_single(self):
        embedder = MockEmbedder(dimension=64)
        embedding = embedder.embed_single("test")

        assert embedding.shape == (64,)

    def test_dimension(self):
        embedder = MockEmbedder(dimension=256)
        assert embedder.dimension == 256


# ════════════════════════════════════════════════════════════════════════
# RUN TESTS
# ════════════════════════════════════════════════════════════════════════

def run_tests():
    print("=" * 50)
    print("Running RAG Tests")
    print("=" * 50)

    all_passed = True

    # TextSplitter tests
    print("\n[TextSplitter]")
    for name in dir(TestTextSplitter):
        if name.startswith("test_"):
            test = getattr(TestTextSplitter(), name)
            try:
                test()
                print(f"  [PASS] {name}")
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                all_passed = False

    # VectorDB tests
    print("\n[VectorDB]")
    for name in dir(TestVectorDB):
        if name.startswith("test_"):
            test = getattr(TestVectorDB(), name)
            try:
                test()
                print(f"  [PASS] {name}")
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                all_passed = False

    # MockEmbedder tests
    print("\n[MockEmbedder]")
    for name in dir(TestMockEmbedder):
        if name.startswith("test_"):
            test = getattr(TestMockEmbedder(), name)
            try:
                test()
                print(f"  [PASS] {name}")
            except Exception as e:
                print(f"  [FAIL] {name}: {e}")
                all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 50)

    return all_passed


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
