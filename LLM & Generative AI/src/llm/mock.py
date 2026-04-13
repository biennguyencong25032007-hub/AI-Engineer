"""
Mock LLM Client - For testing without API keys.
"""
from __future__ import annotations

import time
from typing import AsyncIterator

from src.llm.base import BaseLLMClient, ChatResponse, Message, StreamChunk


class MockLLMClient(BaseLLMClient):
    """Mock LLM client that returns pre-configured responses for testing."""

    def __init__(
        self,
        api_key: str = "mock-key",
        model: str = "mock-model",
        response: str = "This is a mock response.",
        **kwargs,
    ):
        super().__init__(api_key=api_key, model=model, **kwargs)
        self._response = response

    def _init_client(self):
        return self

    def chat(
        self,
        messages: list[Message],
        system: str = None,
        **kwargs,
    ) -> ChatResponse:
        """Return mock response after a short delay."""
        time.sleep(0.1)  # Simulate network delay

        # Build response based on last message
        user_message = ""
        for msg in reversed(messages):
            if msg.role == "user":
                user_message = msg.content
                break

        return ChatResponse(
            content=self._response.format(
                message=user_message,
                model=self.model,
            ),
            model=self.model,
            usage={
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            finish_reason="stop",
        )

    async def achat(
        self,
        messages: list[Message],
        system: str = None,
        **kwargs,
    ) -> ChatResponse:
        """Async version."""
        return self.chat(messages, system, **kwargs)

    def stream(
        self,
        messages: list[Message],
        system: str = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream mock response word by word."""
        response = self.chat(messages, system, **kwargs).content
        for i, word in enumerate(response.split()):
            yield StreamChunk(
                content=word + " ",
                delta=word + " ",
                is_final=(i == len(response.split()) - 1),
            )

    def count_tokens(self, text: str) -> int:
        """Rough estimate: ~4 chars per token."""
        return len(text) // 4
