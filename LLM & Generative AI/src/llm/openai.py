"""
OpenAI Client
Supports GPT-4, GPT-4 Turbo, GPT-3.5 Turbo, và o1-preview.
"""
from __future__ import annotations

from typing import AsyncIterator, Optional

import openai
from openai import AsyncOpenAI, OpenAI

from src.config import config
from src.logger import get_logger
from src.llm.base import BaseLLMClient, ChatResponse, Message, StreamChunk

logger = get_logger("llm.openai")


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI's GPT models."""

    def _init_client(self) -> OpenAI:
        api_key = self.api_key or config.openai_api_key
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")

        kwargs = {"api_key": api_key, "timeout": self.timeout}
        if config.llm.api_base:
            kwargs["base_url"] = config.llm.api_base

        return OpenAI(**kwargs)

    def _init_async_client(self) -> AsyncOpenAI:
        api_key = self.api_key or config.openai_api_key
        kwargs = {"api_key": api_key, "timeout": self.timeout}
        if config.llm.api_base:
            kwargs["base_url"] = config.llm.api_base
        return AsyncOpenAI(**kwargs)

    def chat(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send message to GPT."""
        client: OpenAI = self._client

        # Build message format with system
        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend([
            {"role": m.role, "content": m.content}
            for m in messages
        ])

        response = client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            top_p=kwargs.get("top_p", self.temperature),
            stream=False,
        )

        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            },
            finish_reason=choice.finish_reason,
            raw_response=response,
        )

    async def achat(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        """Async version of chat."""
        client = self._init_async_client()

        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend([
            {"role": m.role, "content": m.content}
            for m in messages
        ])

        response = await client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        choice = response.choices[0]
        return ChatResponse(
            content=choice.message.content or "",
            model=response.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
            },
            finish_reason=choice.finish_reason,
        )

    def stream(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from GPT."""
        client: OpenAI = self._client

        chat_messages = []
        if system:
            chat_messages.append({"role": "system", "content": system})
        chat_messages.extend([
            {"role": m.role, "content": m.content}
            for m in messages
        ])

        stream = client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            stream=True,
        )

        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                delta = chunk.choices[0].delta.content
                yield StreamChunk(content=delta, delta=delta)
            elif chunk.choices and chunk.choices[0].finish_reason:
                yield StreamChunk(content="", is_final=True)

    def count_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Rough estimate: ~4 chars per token
        return len(text) // 4
