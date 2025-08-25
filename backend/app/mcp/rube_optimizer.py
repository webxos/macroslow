from mcp.server import MCPServer
from rube_mcp_client import RubeClient
from backend.app.mcp.curator_server import CuratorServer
from backend.app.mcp.mechanic_server import MechanicServer

class RubeOptimizer(MCPServer):
    def __init__(self):
        super().__init__()
        self.rube = RubeClient(server_url="https://api.rube.app", token="your_rube_token")
        self.curator = CuratorServer()
        self.mechanic = MechanicServer()

    async def optimize_tool_execution(self, tool_name: str, params: dict):
        tools = await self.rube.list_tools()
        if tool_name not in [t["name"] for t in tools]:
            return {"error": "Tool not found"}
        data = await self.curator.fetch_dataset({"dataset_id": "nasa_123"})
        optimized_params = {**params, **{"context": data}}
        result = await self.rube.execute_tool(tool_name, optimized_params)
        await self.mechanic.log_execution(tool_name, result)
        return result

server = RubeOptimizer()
server.run()
