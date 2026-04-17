from .base_agent import BaseAgent

class ReActAgent(BaseAgent):
    def run(self, query: str):
        prompt = f"""
You are a ReAct agent.
Follow format:
Thought -> Action -> Observation -> Final Answer

Question: {query}
"""
        return self.llm(prompt)