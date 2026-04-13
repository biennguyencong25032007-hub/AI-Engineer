"""
Anthropic (Claude) Client
Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku.
"""
from __future__ import annotations

from typing import AsyncIterator, Optional

import anthropic
from anthropic import AsyncAnthropic, Anthropic

from src.config import config
from src.logger import get_logger
from src.llm.base import BaseLLMClient, ChatResponse, Message, StreamChunk

logger = get_logger("llm.anthropic")


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic's Claude models."""

    def _init_client(self) -> Anthropic:
        api_key = self.api_key or config.anthropic_api_key
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        return Anthropic(api_key=api_key, timeout=self.timeout)

    def _init_async_client(self) -> AsyncAnthropic:
        api_key = self.api_key or config.anthropic_api_key
        return AsyncAnthropic(api_key=api_key, timeout=self.timeout)

    def chat(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        **kwargs,
    ) -> ChatResponse:
        """Send message to Claude."""
        client: Anthropic = self._client

        # Build message format
        chat_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        response = client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            system=system or kwargs.get("system", ""),
            messages=chat_messages,
        )

        return ChatResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
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

        chat_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        response = await client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            system=system or kwargs.get("system", ""),
            messages=chat_messages,
        )

        return ChatResponse(
            content=response.content[0].text,
            model=response.model,
            usage={
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            },
            finish_reason=response.stop_reason,
        )

    def stream(
        self,
        messages: list[Message],
        system: Optional[str] = None,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream response from Claude."""
        client: Anthropic = self._client

        chat_messages = [
            {"role": m.role, "content": m.content}
            for m in messages
        ]

        with client.messages.stream(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
            system=system or kwargs.get("system", ""),
            messages=chat_messages,
        ) as stream:
            for event in stream:
                if event.type == "content_block_delta":
                    yield StreamChunk(
                        content=event.delta.text,
                        delta=event.delta.text,
                    )
                elif event.type == "message_delta":
                    yield StreamChunk(content="", is_final=True)

    def count_tokens(self, text: str) -> int:
        """Count tokens using Claude's tokenizer."""
        # Approximate: ~4 chars per token
        return len(text) // 4
