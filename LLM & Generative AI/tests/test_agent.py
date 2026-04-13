"""
Agent Tests
"""
import pytest

from src.agent.base import BaseAgent, AgentState, ConversationAgent
from src.llm.base import BaseLLMClient, ChatResponse, Message


class MockLLM(BaseLLMClient):
    """Mock LLM for testing."""

    def __init__(self):
        pass

    def _init_client(self):
        return None

    def chat(self, messages, system=None, **kwargs):
        return ChatResponse(
            content="Mock response",
            model="mock",
        )

    def achat(self, messages, system=None, **kwargs):
        return self.chat(messages, system)

    def stream(self, messages, system=None, **kwargs):
        yield from []

    def count_tokens(self, text):
        return len(text) // 4


class TestConversationAgent:
    """Test conversation agent."""

    def test_simple_conversation(self):
        llm = MockLLM()
        agent = ConversationAgent(
            llm=llm,
            system_prompt="You are a helpful assistant.",
        )

        response = agent.think("Hello!")
        assert response.content == "Mock response"


class TestAgentState:
    """Test agent states."""

    def test_states(self):
        assert AgentState.IDLE.value == "idle"
        assert AgentState.FINISHED.value == "finished"
        assert AgentState.ERROR.value == "error"