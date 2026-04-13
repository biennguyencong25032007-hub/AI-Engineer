"""
LLM Client Tests
"""
import pytest

from src.llm.base import Message, ChatResponse


class TestMessages:
    """Test message handling."""

    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_message_roles(self):
        for role in ["system", "user", "assistant"]:
            msg = Message(role=role, content="test")
            assert msg.role == role


class TestChatResponse:
    """Test chat response."""

    def test_response_with_usage(self):
        response = ChatResponse(
            content="Hello!",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 5},
        )
        assert response.total_tokens == 15

    def test_response_without_usage(self):
        response = ChatResponse(content="Hello!", model="test-model")
        assert response.total_tokens == 0