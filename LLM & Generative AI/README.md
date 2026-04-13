# GenAI Pipeline 💡

LLM & Generative AI toolkit với **RAG**, **Agents**, và **Tools**.

## 🎯 Features

- **Multi-LLM Support**: Claude (Anthropic), GPT (OpenAI), Groq, Ollama
- **RAG Pipeline**: Embedding, Vector DB (ChromaDB/FAISS), Retrieval
- **Agent Framework**: ReAct, Function Calling, Tool execution
- **Prompt Templates**: Reusable prompts cho common tasks
- **Chain Utilities**: Composable LLM pipelines

## 📦 Install

```bash
cd "Machine Learning/../GenAI"
pip install -e ".[dev]"
```

Copy config:
```bash
cp .env.example .env
# Edit .env với your API keys
```

## 🚀 Quick Start

```python
from src.llm import get_llm
from src.rag import RAGPipeline

# Basic chat
llm = get_llm()
response = llm.chat([Message(role="user", content="Hello!")])
print(response.content)

# RAG
rag = RAGPipeline(llm=llm)
rag.index_text("Your document here", doc_id="doc1")
result = rag.query("What is this about?")
```

## 📁 Project Structure

```
GenAI/
├── src/
│   ├── llm/           # LLM clients
│   ├── rag/           # RAG pipeline
│   ├── agent/         # Agent framework
│   ├── prompts/       # Prompt templates
│   ├── chains/        # Chain utilities
│   ├── config.py     # Configuration
│   └── logger.py     # Logging
├── data/              # Vector DB storage
├── logs/              # Log files
├── notebooks/          # Jupyter notebooks
└── tests/             # Unit tests
```

## 🔧 Configuration

Edit `src/config.py` hoặc environment variables:

```python
# LLM Config
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Model Settings
DEFAULT_MODEL=claude-3-5-sonnet-20241022

# RAG Settings
CHUNK_SIZE=512
TOP_K=5
```

## 🛠️ Usage Examples

### RAG Pipeline
```python
from src.rag import RAGPipeline, ChromaDB

rag = RAGPipeline(llm=llm, vector_db=ChromaDB())

# Index documents
rag.index_file("./docs/guide.md")

# Query
result = rag.query("How do I use this?")
```

### Agent with Tools
```python
from src.agent import ReActAgent
from src.agent.tools import CalculatorTool, SearchTool

tools = [CalculatorTool()]
agent = ReActAgent(llm=llm, tools=tools)
response = agent.think("Calculate 50 * 12")
```

### Prompt Templates
```python
from src.prompts import get_prompt

summarize = get_prompt("summarize")
prompt = summarize.format(text="Your text here...")
```

## 📝 License

MIT