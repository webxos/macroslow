from crewai import Agent, Task, Crew
from pydantic import BaseModel
from qiskit import QuantumCircuit
from django_mongo_engine import connect
from rube_mcp_client import RubeClient
import json

class TaskRequest(BaseModel):
    goal: str
    context: dict = {}

class AlchemistManager:
    def __init__(self):
        connect('webxos_rag')
        self.rube = RubeClient(server_url="https://api.rube.app", token="your_rube_token")
        self.agents = {
            "md_agent": Agent(role="MD Agent", goal="Run molecular simulations", backstory="Specialized in PyTorch MD", llm="openai/gpt-4o"),
            "qc_agent": Agent(role="Quantum Circuit Agent", goal="Optimize quantum circuits", backstory="Claude-based circuit expert", llm="anthropic/claude-3"),
            "kb_agent": Agent(role="Knowledge Base Agent", goal="Manage Django MongoDB", backstory="Database specialist", llm="openai/gpt-4o"),
            "video_agent": Agent(role="Video Scribe", goal="Generate training videos", backstory="Video synthesis expert", llm="openai/gpt-4o")
        }
        self.crew = Crew(agents=list(self.agents.values()), manager_llm="openai/gpt-4o")

    def decompose_task(self, request: TaskRequest):
        if "simulation" in request.goal.lower():
            return Task(description="Run molecular simulation", agent=self.agents["md_agent"], context=request.context)
        elif "quantum circuit" in request.goal.lower():
            return Task(description="Optimize quantum circuit", agent=self.agents["qc_agent"], context=request.context)
        elif "knowledge" in request.goal.lower():
            return Task(description="Update knowledge base", agent=self.agents["kb_agent"], context=request.context)
        else:
            return Task(description="Generate video", agent=self.agents["video_agent"], context=request.context)

    async def manage_workflow(self, request: TaskRequest):
        task = self.decompose_task(request)
        output = await task.execute()
        parsed_data = self.parse_output(output)
        if parsed_data.get("simulation_id"):
            next_task = Task(description="Generate video from report", agent=self.agents["video_agent"], context={"report_url": parsed_data["simulation_id"]})
            video_output = await next_task.execute()
            parsed_data["video_url"] = self.parse_output(video_output).get("video_url")
        if "anomaly" in output.lower():
            await self.rube.execute_tool("github_create_issue", {"title": "Anomaly Detected", "body": output})
        return parsed_data

    def parse_output(self, output):
        try:
            return {"simulation_id": output.split("ID:")[1].split()[0]} if "ID:" in output else {"video_url": output.split("URL:")[1].split()[0]} if "URL:" in output else {}
        except IndexError:
            return {"error": "Parse failed, retrying..."}
            await self.manage_workflow(request)  # Retry logic

manager = AlchemistManager()
