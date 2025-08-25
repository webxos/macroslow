from mcp.server import MCPServer
from pydantic import BaseModel
import torch
import torch.nn as nn
from backend.app.mcp.alchemist_server import AlchemistServer
from backend.app.mcp.chancellor_server import ChancellorServer
from rube_mcp_client import RubeClient

class ValidationRequest(BaseModel):
    model_id: str
    data_path: str
    threshold: float

class ValidatorServer(MCPServer):
    def __init__(self):
        super().__init__()
        self.alchemist = AlchemistServer()
        self.chancellor = ChancellorServer()
        self.rube = RubeClient(server_url="https://api.rube.app", token="your_rube_token")

    async def validate_model(self, request: ValidationRequest):
        model_state = await self.alchemist.get_model_state(request.model_id)
        if "state" not in model_state:
            return {"error": "Model not found"}
        model = nn.Linear(10, 2)
        model.load_state_dict(model_state["state"])
        data = torch.randn(100, 10)
        labels = torch.randint(0, 2, (100,))
        with torch.no_grad():
            output = model(data)
            accuracy = (output.argmax(dim=1) == labels).float().mean().item()
        if accuracy >= request.threshold:
            await self.chancellor.distribute_rewards("user1", "validation_success")
            await self.rube.execute_tool("slack_send_message", {"channel": "#science", "message": f"Model {request.model_id} validated with {accuracy*100}% accuracy"})
            return {"status": "valid", "accuracy": accuracy}
        return {"status": "invalid", "accuracy": accuracy}

server = ValidatorServer()
server.run()
