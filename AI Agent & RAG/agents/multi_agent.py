class MultiAgent:
    def __init__(self, agents: list):
        self.agents = agents

    def run(self, query):
        results = {}
        for agent in self.agents:
            results[agent.__class__.__name__] = agent.run(query)
        return results