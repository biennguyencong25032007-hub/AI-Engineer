"""
Chain Utilities
Composable pipelines for LLM operations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from src.llm.base import BaseLLMClient, Message
from src.logger import get_logger

logger = get_logger("chains")


T = TypeVar("T")
U = TypeVar("U")


@dataclass
class ChainInput:
    """Input to a chain."""
    data: Any


@dataclass
class ChainOutput:
    """Output from a chain."""
    data: Any
    metadata: dict


class BaseChain(ABC, Generic[T, U]):
    """
    Abstract base for chains.

    A chain is a sequence of operations that transform input to output.
    """

    @abstractmethod
    def invoke(self, input: T) -> U:
        """Process input and return output."""
        pass

    def __rshift__(self, other: BaseChain) -> "ChainSequence":
        """Chain two operations with >>."""
        if isinstance(other, ChainSequence):
            return ChainSequence([self] + other.chains)
        return ChainSequence([self, other])

    def __call__(self, input: T) -> U:
        return self.invoke(input)


class ChainSequence(BaseChain):
    """Sequence of chains."""

    def __init__(self, chains: list[BaseChain]):
        self.chains = chains

    def invoke(self, input: T) -> U:
        result = input
        for chain in self.chains:
            result = chain.invoke(result)
        return result

    def __rshift__(self, other: BaseChain) -> "ChainSequence":
        return ChainSequence(self.chains + [other])


# ════════════════════════════════════════════════════════════════════════
# Basic Chains
# ════════════════════════════════════════════════════════════════════════

class PromptChain(BaseChain[str, str]):
    """
    Simple prompt → LLM → response chain.
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        template: str | None = None,
        system: str | None = None,
    ):
        self.llm = llm
        self.template = template
        self.system = system

    def invoke(self, input: str) -> str:
        if self.template:
            prompt = self.template.format(input=input)
        else:
            prompt = input

        response = self.llm.chat(
            messages=[Message(role="user", content=prompt)],
            system=self.system,
        )
        return response.content


class TransformChain(BaseChain[T, U]):
    """
    Generic transformation chain.
    """

    def __init__(self, func: callable):
        self.func = func

    def invoke(self, input: T) -> U:
        return self.func(input)


class ParallelChain(BaseChain[list[T], list[U]]):
    """
    Run multiple chains in parallel.
    """

    def __init__(self, chains: list[BaseChain]):
        self.chains = chains

    def invoke(self, inputs: list[T]) -> list[U]:
        results = []
        for inp, chain in zip(inputs, self.chains):
            results.append(chain.invoke(inp))
        return results


# ════════════════════════════════════════════════════════════════════════
# LLM Chains
# ════════════════════════════════════════════════════════════════════════

class LLMSummarizeChain(TransformChain[str, str]):
    """Summarize text using LLM."""

    def __init__(
        self,
        llm: BaseLLMClient,
        prompt_template: str | None = None,
    ):
        self.llm = llm
        self.prompt_template = prompt_template or """Summarize the following text:

{text}

Summary:"""

        super().__init__(self._summarize)

    def _summarize(self, text: str) -> str:
        prompt = self.prompt_template.format(text=text)
        response = self.llm.chat(
            messages=[Message(role="user", content=prompt)],
        )
        return response.content


class LLMExtractChain(TransformChain[str, dict]):
    """Extract structured data using LLM."""

    def __init__(self, llm: BaseLLMClient, schema: str):
        self.llm = llm
        self.schema = schema

    def invoke(self, text: str) -> dict:
        prompt = f"""Extract information in JSON format based on this schema:

Schema: {self.schema}

Text: {text}

Output JSON:"""

        import json

        response = self.llm.chat(
            messages=[Message(role="user", content=prompt)],
        )

        # Try to parse JSON
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            return {"raw": response.content}


class LLMClassifyChain(TransformChain[str, str]):
    """Classify text using LLM."""

    def __init__(
        self,
        llm: BaseLLMClient,
        categories: list[str],
    ):
        self.llm = llm
        self.categories = categories

    def invoke(self, text: str) -> str:
        prompt = f"""Classify the following text into one of these categories:
{', '.join(self.categories)}

Text: {text}

Category:"""

        response = self.llm.chat(
            messages=[Message(role="user", content=prompt)],
        )
        return response.content.strip()
