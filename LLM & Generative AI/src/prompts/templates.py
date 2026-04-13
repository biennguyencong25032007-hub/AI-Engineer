"""
Prompt Templates
Reusable prompts for common tasks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class PromptTemplate:
    """
    A prompt template with variables.

    Usage:
        template = PromptTemplate(
            template="Hello {name}, you have {count} messages.",
            variables=["name", "count"]
        )
        prompt = template.format(name="Alice", count=5)
    """

    template: str
    variables: list[str]
    description: str = ""

    def format(self, **kwargs) -> str:
        """Format template with variables."""
        # Validate required variables
        for var in self.variables:
            if var not in kwargs:
                raise ValueError(f"Missing required variable: {var}")

        return self.template.format(**kwargs)

    def format_partial(self, **kwargs) -> Callable:
        """Return a function that fills remaining variables."""
        def partial(**remaining):
            return self.format(**{**kwargs, **remaining})
        return partial


# ════════════════════════════════════════════════════════════════════════
# System Prompts
# ════════════════════════════════════════════════════════════════════════

SYSTEM_ASSISTANT = PromptTemplate(
    template="""You are a helpful, harmless, and honest AI assistant.

Guidelines:
- Be clear and concise
- Admit when you don't know
- Provide balanced perspectives
- Be respectful and professional
""",
    variables=[],
    description="Default assistant system prompt"
)

SYSTEM_CODER = PromptTemplate(
    template="""You are an expert Python programmer and software engineer.

Guidelines:
- Write clean, readable, well-documented code
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Include docstrings for functions and classes
- Write unit tests for critical functionality
""",
    variables=[],
    description="System prompt for coding tasks"
)

SYSTEM_SUMMARIZER = PromptTemplate(
    template="""You are an expert at summarizing and extracting key information.

Guidelines:
- Capture the main points accurately
- Use concise language
- Maintain objectivity
- Include key details and figures
- Structure summaries logically
""",
    variables=[],
    description="System prompt for summarization"
)

SYSTEM_ANALYST = PromptTemplate(
    template="""You are a data analyst and business intelligence expert.

Guidelines:
- Focus on actionable insights
- Use clear visualizations descriptions
- Explain methodology
- Acknowledge limitations and uncertainties
- Provide recommendations when appropriate
""",
    variables=[],
    description="System prompt for data analysis"
)

# ════════════════════════════════════════════════════════════════════════
# Task Prompts
# ════════════════════════════════════════════════════════════════════════

PROMPT_SUMMARIZE = PromptTemplate(
    template="""Summarize the following text:

{text}

Summary (include key points and main takeaways):""",
    variables=["text"],
    description="Summarize a text"
)

PROMPT_QUESTION_ANSWER = PromptTemplate(
    template="""Based on the following context, answer the question.

Context:
{context}

Question: {question}

Answer:""",
    variables=["context", "question"],
    description="Answer a question given context"
)

PROMPT_CLASSIFY = PromptTemplate(
    template="""Classify the following text into one of these categories: {categories}

Text: {text}

Category:""",
    variables=["categories", "text"],
    description="Classify text into categories"
)

PROMPT_EXTRACT = PromptTemplate(
    template="""Extract the following information from the text:
{schema}

Text:
{text}

Extracted information:""",
    variables=["schema", "text"],
    description="Extract structured information"
)

PROMPT_TRANSLATE = PromptTemplate(
    template="""Translate the following text to {target_language}:

{text}

Translation:""",
    variables=["target_language", "text"],
    description="Translate text"
)

PROMPT_REWRITE = PromptTemplate(
    template="""Rewrite the following text in a {style} style:

Original text:
{text}

Rewritten text:""",
    variables=["style", "text"],
    description="Rewrite in different style"
)

PROMPT_BRAINSTORM = PromptTemplate(
    template="""Brainstorm ideas for: {topic}

Generate {count} creative and diverse ideas:

""",
    variables=["topic", "count"],
    description="Brainstorm ideas"
)

PROMPT_ANALYZE_CODE = PromptTemplate(
    template="""Analyze the following code and provide:
1. What it does
2. Potential issues or bugs
3. Suggestions for improvement

Code:
```{language}
{code}
```

Analysis:""",
    variables=["language", "code"],
    description="Analyze code"
)

PROMPT_GENERATE_TESTS = PromptTemplate(
    template="""Generate unit tests for the following code:

```{language}
{code}
```

Tests:""",
    variables=["language", "code"],
    description="Generate unit tests"
)

# ════════════════════════════════════════════════════════════════════════
# RAG Prompts
# ════════════════════════════════════════════════════════════════════════

PROMPT_RAG_QA = PromptTemplate(
    template="""You are a helpful assistant. Use the following context to answer the question.

If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Answer:""",
    variables=["context", "question"],
    description="RAG question answering"
)

PROMPT_RAG_SUMMARIZE = PromptTemplate(
    template="""You are a research assistant. Based on the following documents, provide a comprehensive summary.

Documents:
{documents}

Summary:""",
    variables=["documents"],
    description="RAG document summarization"
)


# ════════════════════════════════════════════════════════════════════════
# Prompt Utilities
# ════════════════════════════════════════════════════════════════════════

def get_prompt(name: str) -> PromptTemplate:
    """Get a prompt template by name."""
    prompts = {
        "summarize": PROMPT_SUMMARIZE,
        "qa": PROMPT_QUESTION_ANSWER,
        "classify": PROMPT_CLASSIFY,
        "extract": PROMPT_EXTRACT,
        "translate": PROMPT_TRANSLATE,
        "rewrite": PROMPT_REWRITE,
        "brainstorm": PROMPT_BRAINSTORM,
        "analyze_code": PROMPT_ANALYZE_CODE,
        "generate_tests": PROMPT_GENERATE_TESTS,
        "rag_qa": PROMPT_RAG_QA,
        "rag_summarize": PROMPT_RAG_SUMMARIZE,
    }

    if name not in prompts:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(prompts.keys())}")

    return prompts[name]


def create_prompt(template: str, **kwargs) -> str:
    """Create a prompt from template string."""
    return template.format(**kwargs)
