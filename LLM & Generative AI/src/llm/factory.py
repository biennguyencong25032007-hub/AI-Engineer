"""
LLM Factory - Create LLM clients by provider.
"""
from __future__ import annotations

import os
from typing import Optional, Type

from src.config import config, LLMConfig
from src.llm.base import BaseLLMClient
from src.llm.anthropic import AnthropicClient
from src.llm.openai import OpenAIClient
from src.llm.mock import MockLLMClient
from src.logger import get_logger

logger = get_logger("llm.factory")


class LLMFactory:
    """
    Factory để create LLM clients.

    Usage:
        client = LLMFactory.create(provider="anthropic", model="claude-3-5-sonnet")
        response = client.chat([Message(role="user", content="Hello!")])
    """

    _clients: dict[str, Type[BaseLLMClient]] = {
        "anthropic": AnthropicClient,
        "openai": OpenAIClient,
        "mock": MockLLMClient,
    }

    @classmethod
    def register(cls, name: str, client_class: Type[BaseLLMClient]) -> None:
        """Register a new LLM client class."""
        cls._clients[name] = client_class
        logger.info(f"Registered LLM client: {name}")

    @classmethod
    def create(cls, provider: str | None = None, **kwargs) -> BaseLLMClient:
        """
        Create an LLM client.

        Args:
            provider: LLM provider (anthropic, openai, groq, ollama)
            **kwargs: Override config options

        Returns:
            LLM client instance
        """
        # Use config if provider not specified
        if provider is None:
            provider = config.llm.provider

        # Get client class
        client_class = cls._clients.get(provider)
        if not client_class:
            available = ", ".join(cls._clients.keys())
            raise ValueError(
                f"Unknown provider: {provider}. Available: {available}"
            )

        # Build config kwargs - only pass client-accepted params
        # Client __init__ params: api_key, model, temperature, max_tokens, timeout, max_retries
        llm_config = {
            "api_key": kwargs.get("api_key") or os.getenv(f"{provider.upper()}_API_KEY"),
            "model": kwargs.get("model") or config.llm.model,
            "temperature": kwargs.get("temperature") or config.llm.temperature,
            "max_tokens": kwargs.get("max_tokens") or config.llm.max_tokens,
            "timeout": kwargs.get("timeout") or config.llm.timeout,
            "max_retries": kwargs.get("max_retries") or config.llm.max_retries,
        }
        # Remove None api_key (will use env var or raise error in client)
        if llm_config["api_key"] is None:
            llm_config.pop("api_key")

        logger.debug(f"Creating LLM client: {provider}", model=llm_config.get("model"))

        return client_class(**llm_config)

    @classmethod
    def create_with_fallback(
        cls,
        primary: str = "anthropic",
        fallback: str = "openai",
        **kwargs,
    ) -> BaseLLMClient:
        """
        Create client với automatic fallback.

        If primary fails, tries fallback.
        """
        primary_client = cls.create(primary, **kwargs)

        def _fallback_wrapper(*args, **inner_kwargs):
            try:
                return primary_client.chat(*args, **inner_kwargs)
            except Exception as e:
                logger.warning(
                    f"Primary LLM failed, trying fallback",
                    primary=primary,
                    fallback=fallback,
                    error=str(e),
                )
                fallback_client = cls.create(fallback, **kwargs)
                return fallback_client.chat(*args, **inner_kwargs)

        # Return wrapped client
        return _FallbackLLMClient(primary_client, cls.create(fallback, **kwargs))


class _FallbackLLMClient(BaseLLMClient):
    """Wrapper that tries primary then fallback on failure."""

    def __init__(self, primary: BaseLLMClient, fallback: BaseLLMClient):
        self.primary = primary
        self.fallback = fallback
        self.model = primary.model
        self.temperature = primary.temperature
        self.max_tokens = primary.max_tokens

    def _init_client(self):
        return self.primary._client

    def chat(self, messages, system=None, **kwargs):
        try:
            return self.primary.chat(messages, system, **kwargs)
        except Exception as e:
            logger.warning(f"Primary failed, using fallback", error=str(e))
            return self.fallback.chat(messages, system, **kwargs)

    async def achat(self, messages, system=None, **kwargs):
        try:
            return await self.primary.achat(messages, system, **kwargs)
        except Exception:
            return await self.fallback.achat(messages, system, **kwargs)

    def stream(self, messages, system=None, **kwargs):
        try:
            yield from self.primary.stream(messages, system, **kwargs)
        except Exception:
            yield from self.fallback.stream(messages, system, **kwargs)

    def count_tokens(self, text):
        return self.primary.count_tokens(text)


# Convenience function
def get_llm(provider: str | None = None, **kwargs) -> BaseLLMClient:
    """Quick access to LLM factory."""
    return LLMFactory.create(provider, **kwargs)
