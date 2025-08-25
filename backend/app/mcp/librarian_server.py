from mcp.server import MCPServer
from pydantic import BaseModel
import json

class QueryRequest(BaseModel):
    question: str

class LibrarianServer(MCPServer):
    def __init__(self):
        super().__init__()
        self.knowledge_base = {"connect_telescope": "Use The Astronomer API with NASA hooks."}

    async def query_knowledge_base(self, request: QueryRequest):
        return {"answer": self.knowledge_base.get(request.question, "No answer found")}

    async def log_new_solution(self, solution: dict):
        self.knowledge_base[solution["question"]] = solution["answer"]
        return {"status": "logged"}

    async def get_agent_manifest(self):
        return {
            "agents": [
                {"name": "Curator", "role": "Data Management"},
                {"name": "Sentinel", "role": "Security"},
                {"name": "Chancellor", "role": "Economics"},
                {"name": "Architect", "role": "Templating"},
                {"name": "Mechanic", "role": "DevOps"},
                {"name": "Librarian", "role": "Knowledge"}
            ]
        }

    async def generate_documentation_summary(self, project_path: str):
        return {"summary": f"Project at {project_path} includes data and model files."}

server = LibrarianServer()
server.run()
