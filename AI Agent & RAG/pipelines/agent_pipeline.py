"""Agent Pipeline - Orchestrates agent execution"""

from typing import Dict, Any, List, Optional
from agents.base_agent import BaseAgent
from agents.tool_calling_agent import ToolCallingAgent
from agents.react_agent import ReActAgent
from memory.short_term import ShortTermMemory
from memory.long_term import LongTermMemory
from memory.episodic import EpisodicMemory
from config.settings import Settings
from config.prompts import SYSTEM_PROMPT, AGENT_EXECUTION_PROMPT


class AgentPipeline:
    """Main pipeline for agent-based processing"""

    def __init__(self, agent_type: str = "tool_calling", llm=None, memory_type: str = "short_term"):
        self.settings = Settings()
        self.agent_type = agent_type

        # Initialize memory
        if memory_type == "short_term":
            self.memory = ShortTermMemory()
        elif memory_type == "long_term":
            self.memory = LongTermMemory()
        elif memory_type == "episodic":
            self.memory = EpisodicMemory()
        else:
            self.memory = ShortTermMemory()

        # Initialize LLM
        self.llm = llm or self._default_llm()

        # Initialize agent
        self.agent = self._create_agent()

    def _default_llm(self):
        """Create default LLM based on available API keys"""
        if self.settings.OPENAI_API_KEY:
            from openai import OpenAI
            return lambda prompt: self._call_openai(prompt)
        elif self.settings.ANTHROPIC_API_KEY:
            from anthropic import Anthropic
            return lambda prompt: self._call_anthropic(prompt)
        else:
            # Simple mock LLM for testing
            return lambda prompt: f"Mock response to: {prompt[:100]}..."

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        from openai import OpenAI
        client = OpenAI(api_key=self.settings.OPENAI_API_KEY)

        try:
            response = client.chat.completions.create(
                model=self.settings.DEFAULT_LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"OpenAI API error: {str(e)}"

    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        from anthropic import Anthropic
        client = Anthropic(api_key=self.settings.ANTHROPIC_API_KEY)

        try:
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text
        except Exception as e:
            return f"Anthropic API error: {str(e)}"

    def _create_agent(self) -> BaseAgent:
        """Create agent based on type"""
        if self.agent_type == "tool_calling":
            # Initialize tools
            from tools.calculator_tool import CalculatorTool
            from tools.search_tool import SearchTool
            from tools.file_tool import FileTool
            from tools.sql_tool import SQLTool

            tools = {
                'calculator': CalculatorTool(),
                'search': SearchTool(),
                'file': FileTool(),
                'sql': SQLTool()
            }

            return ToolCallingAgent(self.llm, tools)
        elif self.agent_type == "react":
            return ReActAgent(self.llm)
        else:
            raise ValueError(f"Unknown agent type: {self.agent_type}")

    def run(self, query: str, use_memory: bool = True) -> str:
        """Run query through agent pipeline"""
        try:
            # Build context from memory if enabled
            if use_memory:
                context = self.memory.get_context()
                full_query = f"Context:\n{context}\n\nQuery: {query}" if context else query
            else:
                full_query = query

            # Run agent
            response = self.agent.run(full_query)

            # Store in memory
            if use_memory:
                self.memory.add({
                    'role': 'user',
                    'content': query
                })
                self.memory.add({
                    'role': 'assistant',
                    'content': response
                })

            return response
        except Exception as e:
            return f"Pipeline error: {str(e)}"

    def run_with_retrieval(self, query: str, retriever, k: int = 5) -> str:
        """Run query with RAG retrieval"""
        try:
            # Get relevant context
            context = retriever.get_relevant_documents_text(query, k=k)

            # Combine with query
            full_prompt = AGENT_EXECUTION_PROMPT.format(
                query=query,
                tools=f"Retrieved Context:\n{context}"
            )

            response = self.agent.run(full_prompt)

            # Store in memory
            self.memory.add({
                'role': 'user',
                'content': query,
                'context': context
            })
            self.memory.add({
                'role': 'assistant',
                'content': response
            })

            return response
        except Exception as e:
            return f"Pipeline with retrieval error: {str(e)}"

    def reset_memory(self):
        """Clear agent memory"""
        self.memory.clear()

    def get_memory(self) -> List[Dict[str, Any]]:
        """Get current memory state"""
        if hasattr(self.memory, 'get_recent'):
            return self.memory.get_recent()
        return []

    def save_memory(self, filepath: str):
        """Save memory to file"""
        if hasattr(self.memory, 'to_dict'):
            import json
            data = self.memory.to_dict()
            with open(filepath, 'w') as f:
                json.dump(data, f)

    def load_memory(self, filepath: str):
        """Load memory from file"""
        if hasattr(self.memory, 'from_dict'):
            import json
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.memory = self.memory.from_dict(data)
