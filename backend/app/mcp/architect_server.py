from mcp.server import MCPServer
from pydantic import BaseModel
import torch
import torch.nn as nn
from backend.app.mcp.mechanic_server import MechanicServer

class TrainingRequest(BaseModel):
    model_id: str
    data_path: str
    epochs: int

class AlchemistServer(MCPServer):
    def __init__(self):
        super().__init__()
        self.mechanic = MechanicServer()
        self.models = {}

    async def train_model(self, request: TrainingRequest):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 2)
            def forward(self, x):
                return self.fc(x)
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        data = torch.randn(100, 10)  # Placeholder data
        labels = torch.randint(0, 2, (100,))  # Placeholder labels
        for epoch in range(request.epochs):
            optimizer.zero_grad()
            output = model(data)
            loss = nn.functional.cross_entropy(output, labels)
            loss.backward()
            optimizer.step()
        self.models[request.model_id] = model.state_dict()
        await self.mechanic.orchestrate_training_job(request)
        return {"model_id": request.model_id, "status": "trained"}

    async def get_model_state(self, model_id: str):
        return {"state": self.models.get(model_id, "Model not found")}

server = AlchemistServer()
server.run()
