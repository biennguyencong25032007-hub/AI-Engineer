import json

class ToolCallingAgent(BaseAgent):
    def __init__(self, llm, tools: dict):
        super().__init__(llm)
        self.tools = tools

    def run(self, query: str):
        prompt = f"""
You can call tools.
Return JSON:
{{"tool": "name", "input": "..."}}

Query: {query}
"""
        response = self.llm(prompt)

        try:
            data = json.loads(response)
            tool = data.get("tool")
            inp = data.get("input")
            if tool in self.tools:
                return self.tools[tool](inp)
        except:
            pass

        return response