from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import aiohttp
import yaml
import json
import time
from datetime import datetime
from src.glastonbury_2048.mcp_server import GlastonburyQuantumOrchestrator
from src.glastonbury_2048.aes_2048 import AES2048Encryptor

# Team Instruction: Implement backend server for INFINITY UI, integrating with GLASTONBURY 2048.
# Use RAG to treat API data as temporary context, with OAuth for anonymous access.
app = FastAPI(title="INFINITY UI Backend")

class SyncRequest(BaseModel):
    api_endpoint: str

class Config:
    def __init__(self):
        with open("infinity_config.yaml", "r") as f:
            self.config = yaml.safe_load(f)
        self.data_cache = []
        self.encryptor = AES2048Encryptor()
        self.orchestrator = GlastonburyQuantumOrchestrator()

async def fetch_api_data(api_endpoint: str) -> list:
    """Fetches data from the specified API endpoint with OAuth."""
    async with aiohttp.ClientSession() as session:
        headers = {"Authorization": f"Bearer {Config().config['oauth_token']}"}
        async with session.get(api_endpoint, headers=headers) as response:
            if response.status != 200:
                raise HTTPException(status_code=response.status, detail="API fetch failed")
            return await response.json()

@app.get("/config")
async def get_config():
    """Returns the current configuration."""
    return Config().config

@app.post("/sync")
async def sync_data(request: SyncRequest):
    """Syncs API data to cache using RAG, updating every 30 seconds."""
    config = Config()
    data = await fetch_api_data(request.api_endpoint)
    config.data_cache = data  # Store in temporary context for RAG
    encrypted_data = config.encryptor.encrypt(json.dumps(data).encode())
    
    # Integrate with GLASTONBURY 2048 for processing
    node_signals = {"node_0": True, "node_1": True, "node_2": True, "node_3": True}
    processed_data, _ = await config.orchestrator.execute_workflow(
        "workflows/infinity_workflow.maml.md", 
        {"data": encrypted_data.hex()}, 
        node_signals, 
        config.config["neuralink_stream"],
        config.config["donor_wallet_id"]
    )
    return {"count": len(processed_data), "timestamp": datetime.now().isoformat()}

@app.post("/export")
async def export_data():
    """Exports cached data to a MAML file."""
    config = Config()
    if not config.data_cache:
        raise HTTPException(status_code=400, detail="No data to export")
    
    maml_data = {
        "maml_version": "2.0",
        "id": f"urn:uuid:{str(time.time()).replace('.', '-')}",
        "type": "api-data-export",
        "data": config.data_cache,
        "timestamp": datetime.now().isoformat()
    }
    output_file = f"workflows/infinity_export_{int(time.time())}.maml.md"
    with open(output_file, "w") as f:
        f.write(f"---\n{yaml.dump(maml_data, sort_keys=False)}\n---\n# API Data Export\n\nExported data from {config.config['api_endpoint']} at {maml_data['timestamp']}")
    return {"file": output_file, "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)