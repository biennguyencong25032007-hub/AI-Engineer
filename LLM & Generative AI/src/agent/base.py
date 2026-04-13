"""
Agent Base Classes
Core agent architecture with memory, reasoning, và tool use.
"""
from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from src.llm.base import BaseLLMClient, Message
from src.logger import get_logger

logger = get_logger("agent.base")


class AgentState(Enum):
    """Agent execution states."""
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING = "waiting"
    FINISHED = "finished"
    ERROR = "error"


@dataclass
class Message:
    """Agent message."""
    role: str
    content: str
    tool_calls: list[dict] | None = None
    tool_results: list[dict] | None = None


@dataclass
class AgentResponse:
    """Response from agent."""
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    state: AgentState = AgentState.FINISHED
    iterations: int = 0


class BaseAgent(ABC):
    """
    Abstract base class for agents.

    Implements:
    - Message history management
    - Tool execution
    - Iterative reasoning
    - Error handling
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        system_prompt: str,
        tools: list[Any] | None = None,
        max_iterations: int = 10,
        verbose: bool = False,
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.max_iterations = max_iterations
        self.verbose = verbose

        # State
        self.messages: list[Message] = []
        self.state = AgentState.IDLE
        self.iteration = 0

        # Tool registry
        self._tool_map = {tool.name: tool for tool in self.tools}

        logger.info(f"Agent initialized: {self.__class__.__name__}")

    @abstractmethod
    def think(self, user_input: str) -> AgentResponse:
        """
        Main agent loop.

        Returns:
            AgentResponse with final output
        """
        pass

    def add_message(self, role: str, content: str) -> None:
        """Add message to history."""
        self.messages.append(Message(role=role, content=content))

    def _format_messages(self) -> list[dict]:
        """Format messages for LLM."""
        formatted = [{"role": "system", "content": self.system_prompt}]

        for msg in self.messages:
            formatted.append({"role": msg.role, "content": msg.content})

        return formatted

    def execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool by name."""
        tool = self._tool_map.get(name)
        if not tool:
            return f"Error: Unknown tool '{name}'"

        valid, error = tool.validate_args(arguments)
        if not valid:
            return f"Error: {error}"

        result = tool.execute(**arguments)

        if result.success:
            return f"Tool result: {result.output}"
        else:
            return f"Tool error: {result.error}"

    def _extract_tool_calls(self, content: str) -> list[dict]:
        """Extract tool calls from LLM response."""
        # Try JSON format
        tool_calls = []

        # Look for tool call patterns
        patterns = [
            r'<tool_call>\s*(\{[^}]+\})\s*</tool_call>',
            r'"tool_calls":\s*\[([^\]]+)\]',
            r'Tool:\s*(\w+)\s+Args:\s*(\{[^}]+\})',
        ]

        for pattern in patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    if "{" in match.group(1):
                        data = json.loads(match.group(1))
                    else:
                        data = {
                            "name": match.group(1).strip(),
                            "arguments": json.loads(match.group(2)) if len(match.groups()) > 1 else {},
                        }
                    tool_calls.append(data)
                except json.JSONDecodeError:
                    continue

        return tool_calls


class ReActAgent(BaseAgent):
    """
    ReAct (Reasoning + Acting) Agent.

    Implements the ReAct prompting pattern:
    - Thought: What should I do?
    - Action: Execute a tool
    - Observation: See the result
    - Repeat until done
    """

    def think(self, user_input: str) -> AgentResponse:
        """Run ReAct reasoning loop."""
        self.add_message("user", user_input)
        self.state = AgentState.THINKING

        for self.iteration in range(1, self.max_iterations + 1):
            # Get LLM response
            messages = self._format_messages()

            response = self.llm.chat(
                messages=[Message(role="user", content=m.content) for m in self.messages[1:]],
                system=self.system_prompt + self._get_tool_instructions(),
            )

            self.add_message("assistant", response.content)

            if self.verbose:
                logger.info(f"[Iteration {self.iteration}] {response.content[:200]}...")

            # Check for final answer
            if "FINAL ANSWER:" in response.content or "Answer:" in response.content:
                self.state = AgentState.FINISHED
                return AgentResponse(
                    content=response.content,
                    iterations=self.iteration,
                )

            # Extract and execute tools
            tool_calls = self._extract_tool_calls(response.content)

            for tool_call in tool_calls:
                tool_name = tool_call.get("name", "")
                tool_args = tool_call.get("arguments", {})

                if tool_name in self._tool_map:
                    result = self.execute_tool(tool_name, tool_args)
                    self.add_message("user", f"Tool result: {result}")

        self.state = AgentState.FINISHED
        return AgentResponse(
            content=self.messages[-1].content,
            iterations=self.iteration,
        )

    def _get_tool_instructions(self) -> str:
        """Get tool usage instructions for system prompt."""
        if not self.tools:
            return ""

        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}"
            for t in self.tools
        ])

        return f"""

You have access to these tools:
{tool_descriptions}

Use tools when needed. Format tool calls as:
<tool_call>
{{"name": "tool_name", "arguments": {{"arg1": "value1"}}}}
</tool_call>

When you have the answer, start with "FINAL ANSWER:" """


class FunctionCallingAgent(BaseAgent):
    """
    Function Calling Agent.

    Uses native function calling support from modern LLMs.
    """

    def think(self, user_input: str) -> AgentResponse:
        """Run function calling loop."""
        self.add_message("user", user_input)
        self.state = AgentState.THINKING

        for self.iteration in range(1, self.max_iterations + 1):
            # Build messages
            messages = self._format_messages()

            # Get response with tool schemas
            response = self.llm.chat(
                messages=[Message(role="user", content=m.content) for m in self.messages[1:]],
                system=self.system_prompt,
            )

            self.add_message("assistant", response.content)

            if self.verbose:
                logger.info(f"[Iteration {self.iteration}]")

            # Check for final response
            # (Function calling models usually don't include final text with tool calls)
            if not response.raw_response or not hasattr(response.raw_response, 'tool_calls'):
                self.state = AgentState.FINISHED
                return AgentResponse(
                    content=response.content,
                    iterations=self.iteration,
                )

            # Execute tool calls
            tool_calls = response.raw_response.tool_calls
            if not tool_calls:
                self.state = AgentState.FINISHED
                return AgentResponse(
                    content=response.content,
                    iterations=self.iteration,
                )

            # Execute each tool
            for tool_call in tool_calls:
                func_name = tool_call.function.name
                func_args = json.loads(tool_call.function.arguments)

                result = self.execute_tool(func_name, func_args)
                self.add_message("tool", f"{func_name}: {result}")

        self.state = AgentState.FINISHED
        return AgentResponse(
            content=self.messages[-1].content,
            iterations=self.iteration,
        )


class ConversationAgent(BaseAgent):
    """Simple conversational agent without tool use."""

    def think(self, user_input: str) -> AgentResponse:
        """Simple chat response."""
        self.add_message("user", user_input)

        response = self.llm.chat(
            messages=[Message(role="user", content=m.content) for m in self.messages[1:]],
            system=self.system_prompt,
        )

        self.add_message("assistant", response.content)

        return AgentResponse(content=response.content)
