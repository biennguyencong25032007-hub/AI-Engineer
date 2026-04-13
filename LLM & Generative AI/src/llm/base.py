"""
Base LLM Client Interface
Abstract interface cho tất cả LLM providers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator, Literal, Optional

from src.logger import get_logger

logger = get_logger("llm.base")


@dataclass
class Message:
    """Chat message."""
    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None


@dataclass
class ChatResponse:
    """Standardized LLM response."""
    content: str
    model: str
    usage: dict = field(default_factory=dict)
    finish_reason: Optional[str] = None
    raw_response: Optional[object] = None

    @property
    def total_tokens(self) -> int:
        return self.usage.get("total_tokens", 0)

    @property
    def prompt_tokens(self) -> int:
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        return self.usage.get("completion_tokens", 0)


@dataclass
class StreamChunk:
    """Streaming response chunk."""
    content: str
    delta: str = ""
    is_final: bool = False


class BaseLLMClient(ABC):
    """
    Abstract base class cho LLM clients.

    Implementations:
    - AnthropicClient (Claude)
    - OpenAIClient (GPT-4, GPT-3.5)
    - GroqClient (Llama, Mixtral)
    - OllamaClient (local models)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        timeout: int = 60,
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self._client = self._init_client()

    @abstractmethod
    def _init_client(self) -> object:
        """Initialize the underlying HTTP/client object."""
        pass

    @abstractmethod
    def chat(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        """
        Send chat request và return response.

        Args:
            messages: List of conversation messages
            system: Optional system prompt
            **kwargs: Provider-specific options

        Returns:
            ChatResponse object
        """
        pass

    @abstractmethod
    async def achat(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        """Async version of chat."""
        pass

    @abstractmethod
    def stream(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """
        Stream response chunks.

        Yields:
            StreamChunk objects
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        pass

    def chat_with_retries(
        self,
        messages: list[Message],
        system: Optional[str] = None,
    ) -> ChatResponse:
        """Chat với automatic retries."""
        import time
        import httpx

        last_error = None
        for attempt in range(self.max_retries):
            try:
                return self.chat(messages, system)
            except (httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(
                    f"LLM request failed (attempt {attempt + 1}), retrying in {wait_time}s",
                    error=str(e),
                )
                time.sleep(wait_time)

        raise RuntimeError(f"LLM request failed after {self.max_retries} retries: {last_error}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model})"
