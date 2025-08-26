from crewai import Agent, Task, Crew
from backend.app.database import MongoDBClient
from typing import List, Dict, Any

class CrewAIOrchestration:
    def __init__(self):
        self.db = MongoDBClient()
        self.crew = self._setup_crew()

    def _setup_crew(self) -> Crew:
        curator = Agent(role="Curator", goal="Validate MAML data")
        sentinel = Agent(role="Sentinel", goal="Secure MAML operations")
        alchemist = Agent(role="Alchemist", goal="Execute MAML code")
        return Crew(agents=[curator, sentinel, alchemist], tasks=[])

    def orchestrate_maml(self, maml_id: str) -> Dict[str, Any]:
        maml_data = self.db.get_maml(maml_id)
        if not maml_data:
            return {"error": "MAML not found"}

        tasks = [
            Task(description="Validate MAML structure", agent=self.crew.agents[0]),
            Task(description="Secure execution environment", agent=self.crew.agents[1]),
            Task(description="Execute code blocks", agent=self.crew.agents[2])
        ]
        self.crew.tasks = tasks
        result = self.crew.kickoff()
        return {"status": "completed", "result": result}
