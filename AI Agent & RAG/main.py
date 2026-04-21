"""Main entry point for AI Agent & RAG system"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import Settings
from pipelines.agent_pipeline import AgentPipeline
from pipelines.rag_pipeline import RAGPipeline
from memory import ShortTermMemory


def demo_agent():
    """Demo agent pipeline"""
    print("\n" + "=" * 50)
    print("AGENT PIPELINE DEMO")
    print("=" * 50)

    try:
        # Initialize pipeline
        pipeline = AgentPipeline(agent_type="tool_calling", memory_type="short_term")

        # Run some queries
        queries = [
            "What is 25 * 4?",
            "Calculate 150 / 3",
            "What is the square root of 144?"
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            response = pipeline.run(query)
            print(f"Response: {response}")

        print("\n✓ Agent demo completed!")

    except Exception as e:
        print(f"\n✗ Agent demo failed: {str(e)}")


def demo_rag():
    """Demo RAG pipeline"""
    print("\n" + "=" * 50)
    print("RAG PIPELINE DEMO")
    print("=" * 50)

    try:
        # Initialize pipeline
        pipeline = RAGPipeline()

        # Create sample documents if none exist
        docs_dir = Path("./data/documents")
        docs_dir.mkdir(parents=True, exist_ok=True)

        sample_file = docs_dir / "sample.txt"
        if not sample_file.exists():
            sample_content = """
            Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines.
            Machine Learning is a subset of AI that enables systems to learn and improve from experience.
            Deep Learning is a type of machine learning inspired by the structure of the human brain.
            Natural Language Processing (NLP) is a field of AI focused on enabling computers to understand human language.
            Computer Vision is an AI field that trains computers to interpret visual data from the world.
            """
            sample_file.write_text(sample_content)

        # Index documents
        print("\nIndexing documents...")
        pipeline.index_documents()

        # Run queries
        queries = [
            "What is machine learning?",
            "How is deep learning related to AI?",
            "What are applications of AI?"
        ]

        for query in queries:
            print(f"\nQuery: {query}")
            result = pipeline.query(query, k=3)

            print(f"Answer: {result['answer'][:200]}...")
            print(f"Sources used: {len(result['sources'])}")

        print("\n✓ RAG demo completed!")

    except Exception as e:
        print(f"\n✗ RAG demo failed: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point"""
    print("=" * 50)
    print("AI AGENT & RAG SYSTEM")
    print("=" * 50)

    # Check settings
    try:
        settings = Settings()
        print("\nConfiguration loaded:")
        print(f"  LLM Model: {settings.DEFAULT_LLM_MODEL}")
        print(f"  Embedding Model: {settings.DEFAULT_EMBEDDING_MODEL}")
        print(f"  Vector Store: {settings.VECTOR_STORE_TYPE}")
    except Exception as e:
        print(f"\nWarning: {str(e)}")
        print("Using default configuration...")

    # Run demos
    demo_agent()
    demo_rag()

    print("\n" + "=" * 50)
    print("All demos completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
