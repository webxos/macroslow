from mcp.server import MCPServer
from pydantic import BaseModel
import requests
import json

class SpaceDataRequest(BaseModel):
    dataset: str
    start_date: str
    end_date: str

class AstronomerServer(MCPServer):
    def __init__(self):
        super().__init__()
        self.nasa_api_key = "your_nasa_api_key"
        self.base_url = "https://api.nasa.gov"

    async def fetch_space_data(self, request: SpaceDataRequest):
        params = {
            "api_key": self.nasa_api_key,
            "start_date": request.start_date,
            "end_date": request.end_date
        }
        response = requests.get(f"{self.base_url}/planetary/apod", params=params)
        if response.status_code == 200:
            return {"data": response.json()}
        return {"error": "Data fetch failed"}

    async def process_telescope_data(self, dataset_id: str):
        # Simulate processing telescope data from GIBS
        url = f"{self.base_url}/gibs/{dataset_id}"
        response = requests.get(url, params={"api_key": self.nasa_api_key})
        if response.status_code == 200:
            return {"processed_data": response.json()}
        return {"error": "Processing failed"}

server = AstronomerServer()
server.run()
