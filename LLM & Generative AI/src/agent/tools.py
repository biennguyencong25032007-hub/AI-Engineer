"""
Agent Tool System
Define và execute tools for agents.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Type

from src.logger import get_logger

logger = get_logger("tools")


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    output: Any
    error: str | None = None
    metadata: dict = field(default_factory=dict)


class BaseTool(ABC):
    """
    Abstract base for agent tools.

    Tools allow agents to interact with external systems.
    """

    name: str = ""
    description: str = ""
    parameters: dict = {}

    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Execute the tool with given arguments."""
        pass

    def validate_args(self, args: dict) -> tuple[bool, str | None]:
        """Validate tool arguments."""
        required = [p["name"] for p in self.parameters.get("required", [])]

        for req in required:
            if req not in args:
                return False, f"Missing required argument: {req}"

        return True, None

    def __repr__(self) -> str:
        return f"Tool({self.name})"


# ════════════════════════════════════════════════════════════════════════
# Built-in Tools
# ════════════════════════════════════════════════════════════════════════

class SearchTool(BaseTool):
    """Search tool for RAG pipelines."""

    name = "search"
    description = "Search through indexed documents for information."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query"
            },
            "top_k": {
                "type": "integer",
                "description": "Number of results to return",
                "default": 5
            }
        },
        "required": ["query"]
    }

    def __init__(self, rag_pipeline):
        self.rag = rag_pipeline

    def execute(self, query: str, top_k: int = 5) -> ToolResult:
        try:
            results = self.rag.retrieve(query, top_k=top_k)
            output = "\n".join([
                f"[{r.score:.2f}] {r.text[:200]}..."
                for r in results.chunks
            ])
            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class CalculatorTool(BaseTool):
    """Mathematical calculator tool."""

    name = "calculate"
    description = "Perform mathematical calculations."
    parameters = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression (e.g., '2 + 2', 'sqrt(16)')"
            }
        },
        "required": ["expression"]
    }

    def execute(self, expression: str) -> ToolResult:
        try:
            import math

            # Safe evaluation with math functions
            safe_dict = {"__builtins__": {}, "math": math}
            result = eval(expression, safe_dict)

            return ToolResult(
                success=True,
                output=f"{expression} = {result}",
                metadata={"result": result}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Calculation error: {str(e)}"
            )


class WebSearchTool(BaseTool):
    """Web search tool (using DuckDuckGo or similar)."""

    name = "web_search"
    description = "Search the web for current information."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results",
                "default": 5
            }
        },
        "required": ["query"]
    }

    def __init__(self):
        self._available = False
        try:
            from duckduckgo_search import DDGS
            self._ddgs = DDGS()
            self._available = True
        except ImportError:
            logger.warning("duckduckgo-search not installed, web search unavailable")

    def execute(self, query: str, num_results: int = 5) -> ToolResult:
        if not self._available:
            return ToolResult(
                success=False,
                output=None,
                error="Web search not available. Install: pip install duckduckgo-search"
            )

        try:
            results = list(self._ddgs.text(query, max_results=num_results))

            if not results:
                return ToolResult(success=True, output="No results found.")

            output = "\n".join([
                f"- {r['title']}: {r['href']}"
                for r in results
            ])

            return ToolResult(
                success=True,
                output=output,
                metadata={"num_results": len(results)}
            )
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class FileReadTool(BaseTool):
    """Tool to read files."""

    name = "read_file"
    description = "Read the contents of a file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to read"
            },
            "max_lines": {
                "type": "integer",
                "description": "Maximum number of lines to read",
                "default": 100
            }
        },
        "required": ["path"]
    }

    def execute(self, path: str, max_lines: int = 100) -> ToolResult:
        try:
            from pathlib import Path

            file_path = Path(path)
            if not file_path.exists():
                return ToolResult(success=False, output=None, error="File not found")

            content = file_path.read_text(encoding="utf-8")
            lines = content.split("\n")

            if len(lines) > max_lines:
                content = "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"

            return ToolResult(success=True, output=content)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class PythonExecuteTool(BaseTool):
    """Execute Python code."""

    name = "execute_python"
    description = "Execute Python code and return the result."
    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute"
            }
        },
        "required": ["code"]
    }

    def execute(self, code: str) -> ToolResult:
        try:
            import io
            import sys

            # Capture output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = io.StringIO()
            sys.stderr = io.StringIO()

            result = exec(code, {"__builtins__": __builtins__})

            stdout = sys.stdout.getvalue()
            stderr = sys.stderr.getvalue()

            sys.stdout = old_stdout
            sys.stderr = old_stderr

            output = stdout if stdout else "Code executed (no output)"
            if stderr:
                output += f"\nErrors:\n{stderr}"

            return ToolResult(success=True, output=output)
        except Exception as e:
            return ToolResult(success=False, output=None, error=str(e))


class ToolRegistry:
    """Registry for available tools."""

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def unregister(self, name: str) -> None:
        """Unregister a tool."""
        self._tools.pop(name, None)

    def get(self, name: str) -> BaseTool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[dict]:
        """List all available tools."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "parameters": t.parameters,
            }
            for t in self._tools.values()
        ]

    @property
    def tool_schemas(self) -> list[dict]:
        """Get tool schemas for function calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            }
            for t in self._tools.values()
        ]
