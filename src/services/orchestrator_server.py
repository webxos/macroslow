import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from oqs import Signature
import requests
import json
import uuid

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class OrchestrationPayload(BaseModel):
    maml_data: str
    oauth_token: str
    security_mode: str
    wallet_address: str
    reputation: int

class OrchestrationResponse(BaseModel):
    task_id: str
    status: str
    dunes_hash: str
    signature: str

@app.post("/api/services/orchestrator", response_model=OrchestrationResponse)
async def orchestrate_task(payload: OrchestrationPayload):
    """
    Orchestrate tasks across Vial agents with DUNES security.
    
    Args:
        payload (OrchestrationPayload): MAML data, OAuth token, security mode, wallet, and reputation.
    
    Returns:
        OrchestrationResponse: Task ID, status, DUNES hash, and signature.
    """
    try:
        # Validate OAuth token
        headers = {"Authorization": f"Bearer {payload.oauth_token}"}
        auth_response = requests.post(
            "https://webxos.auth.us-east-1.amazoncognito.com/oauth2/token",
            headers=headers
        )
        auth_response.raise_for_status()
        
        # Parse and validate .MAML.ml
        parse_response = requests.post(
            "http://localhost:8000/api/services/maml_ml_parser",
            json={"mamlData": payload.maml_data, "signature": ""}
        )
        parse_response.raise_for_status()
        parsed_data = parse_response.json()
        if not parsed_data['valid']:
            raise ValueError("Invalid .MAML.ml data
