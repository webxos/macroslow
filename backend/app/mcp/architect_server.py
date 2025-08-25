from mcp.server import MCPServer
from pydantic import BaseModel
import os
import shutil

class TemplateRequest(BaseModel):
    template_name: str
    user_id: str

class ArchitectServer(MCPServer):
    def __init__(self):
        super().__init__()
        self.templates = {
            "data-scientist-vial": "/templates/data-scientist",
            "astronomer-vial": "/templates/astronomer",
            "validator-vial": "/templates/validator",
            "gateway-vial": "/templates/gateway"
        }

    async def list_available_templates(self):
        return {"templates": list(self.templates.keys())}

    async def instantiate_template(self, request: TemplateRequest):
        if request.template_name not in self.templates:
            return {"error": "Template not found"}
        src = self.templates[request.template_name]
        dest = f"/projects/{request.user_id}_{request.template_name}"
        shutil.copytree(src, dest)
        with open(f"{dest}/config.json", "w") as f:
            f.write(json.dumps({"user_id": request.user_id, "api_key": "your_api_key"}))
        return {"project_path": dest}

    async def get_template_readme(self, template_name: str):
        if template_name in self.templates:
            with open(f"{self.templates[template_name]}/README.md", "r") as f:
                return {"readme": f.read()}
        return {"error": "Template not found"}

    async def validate_project_config(self, project_path: str):
        if os.path.exists(f"{project_path}/config.json"):
            return {"status": "valid"}
        return {"status": "invalid", "error": "Config file missing"}

server = ArchitectServer()
server.run()
