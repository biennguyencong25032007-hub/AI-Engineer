"""
GenAI Pipeline - Main Entry Point
Examples và CLI cho LLM & Generative AI project.
"""
from __future__ import annotations

from pathlib import Path

from src.config import config
from src.llm import get_llm, AnthropicClient, OpenAIClient
from src.llm.base import Message
from src.rag import RAGPipeline, TextSplitter
from src.agent import create_agent, create_rag_agent, ReActAgent
from src.prompts import get_prompt, create_prompt
from src.logger import setup_logger

logger = setup_logger("main")


# ════════════════════════════════════════════════════════════════════════
# Examples
# ════════════════════════════════════════════════════════════════════════

def example_basic_chat():
    """Basic chat example với Claude."""
    logger.info("=== Basic Chat Example ===")

    # Use mock for testing (no API key needed)
    client = get_llm(provider="mock")
    response = client.chat([
        Message(
            role="user",
            content="Hello! What can you do?"
        )
    ])

    print(f"Response: {response.content}")
    print(f"Tokens used: {response.total_tokens}")


def example_rag_pipeline():
    """RAG pipeline example."""
    logger.info("=== RAG Pipeline Example ===")

    # Initialize LLM (use mock for testing)
    llm = get_llm(provider="mock")

    # Create RAG pipeline
    rag = RAGPipeline(llm=llm)

    # Index some documents
    docs = [
        ("Python is a high-level programming language.", {"source": "python"}),
        ("Machine learning is a subset of AI.", {"source": "ml"}),
        ("Deep learning uses neural networks.", {"source": "dl"}),
    ]

    for text, meta in docs:
        rag.index_text(text, doc_id=meta["source"], metadata=meta)

    # Query
    result = rag.query("What is Python?")
    print(f"Answer: {result['answer']}")
    print(f"Sources: {len(result['sources'])}")


def example_agent_with_tools():
    """Agent with tool execution."""
    logger.info("=== Agent Example ===")

    from src.agent.tools import CalculatorTool, SearchTool

    # Use mock for testing (no API key needed)
    llm = get_llm(provider="mock")
    tools = [
        CalculatorTool(),
    ]

    agent = ReActAgent(
        llm=llm,
        system_prompt="You are a helpful assistant.",
        tools=tools,
        verbose=True,
    )

    response = agent.think("What is 15 * 23?")
    print(f"Response: {response.content}")


def example_prompt_templates():
    """Prompt template examples."""
    logger.info("=== Prompt Templates Example ===")

    # Get built-in template
    summarize = get_prompt("summarize")
    text = """
    Artificial intelligence (AI) is intelligence demonstrated by machines,
    in contrast to the natural intelligence displayed by humans and animals.
    """
    prompt = summarize.format(text=text)
    print(f"Prompt:\n{prompt}")


# ════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════

def main():
    """Run examples."""
    import argparse

    parser = argparse.ArgumentParser(description="GenAI Pipeline")
    parser.add_argument("example", nargs="?", default="chat", choices=[
        "chat", "rag", "agent", "prompts", "all"
    ])
    args = parser.parse_args()

    if args.example in ("chat", "all"):
        example_basic_chat()

    if args.example in ("rag", "all"):
        example_rag_pipeline()

    if args.example in ("agent", "all"):
        example_agent_with_tools()

    if args.example in ("prompts", "all"):
        example_prompt_templates()


if __name__ == "__main__":
    main()