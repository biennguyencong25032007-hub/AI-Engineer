"""
Agent Factory - Create agents by type.
"""
from __future__ import annotations

from typing import Optional

from src.config import config
from src.llm.base import BaseLLMClient
from src.agent.base import BaseAgent, ReActAgent, FunctionCallingAgent, ConversationAgent
from src.agent.tools import ToolRegistry, BaseTool, SearchTool, CalculatorTool
from src.rag.pipeline import RAGPipeline
from src.logger import get_logger

logger = get_logger("agent.factory")


class AgentFactory:
    """Factory for creating agent instances."""

    @staticmethod
    def create(
        agent_type: str = "conversational",
        llm: Optional[BaseLLMClient] = None,
        tools: list[BaseTool] | None = None,
        system_prompt: str | None = None,
        **kwargs,
    ) -> BaseAgent:
        """
        Create an agent by type.

        Args:
            agent_type: Type of agent (conversational, react, function_calling)
            llm: LLM client instance
            tools: List of tools
            system_prompt: Custom system prompt
            **kwargs: Additional agent config

        Returns:
            Agent instance
        """
        from src.llm.factory import get_llm

        if llm is None:
            llm = get_llm()

        if tools is None:
            tools = []

        if system_prompt is None:
            system_prompt = config.agent.system_prompt

        # Create agent based on type
        if agent_type == "react":
            agent = ReActAgent(
                llm=llm,
                system_prompt=system_prompt,
                tools=tools,
                max_iterations=kwargs.get("max_iterations", config.agent.max_iterations),
                verbose=kwargs.get("verbose", config.agent.verbose),
            )
        elif agent_type == "function_calling":
            agent = FunctionCallingAgent(
                llm=llm,
                system_prompt=system_prompt,
                tools=tools,
                max_iterations=kwargs.get("max_iterations", config.agent.max_iterations),
                verbose=kwargs.get("verbose", config.agent.verbose),
            )
        else:
            agent = ConversationAgent(
                llm=llm,
                system_prompt=system_prompt,
            )

        logger.info(f"Created agent: {agent_type}")
        return agent

    @staticmethod
    def create_with_rag(
        llm: Optional[BaseLLMClient] = None,
        rag_pipeline: Optional[RAGPipeline] = None,
        agent_type: str = "react",
        **kwargs,
    ) -> BaseAgent:
        """Create agent with RAG search tool."""
        from src.llm.factory import get_llm

        if llm is None:
            llm = get_llm()
        if rag_pipeline is None:
            rag_pipeline = RAGPipeline(llm=llm)

        tools = [SearchTool(rag_pipeline)]
        system_prompt = kwargs.pop(
            "system_prompt",
            "You are a helpful assistant with access to a knowledge base."
        )

        return AgentFactory.create(
            agent_type=agent_type,
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
            **kwargs,
        )


# Convenience functions
def create_agent(
    agent_type: str = "conversational",
    **kwargs,
) -> BaseAgent:
    """Quick access to agent creation."""
    return AgentFactory.create(agent_type, **kwargs)


def create_rag_agent(
    llm: Optional[BaseLLMClient] = None,
    rag_pipeline: Optional[RAGPipeline] = None,
    **kwargs,
) -> BaseAgent:
    """Create RAG-enabled agent."""
    return AgentFactory.create_with_rag(llm, rag_pipeline, **kwargs)
