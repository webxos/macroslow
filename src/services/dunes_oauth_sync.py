import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import jwt

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class SyncPayload(BaseModel):
    oauthToken: str
    mamlData: str
    dunesHash: str
    signature: str

class SyncResponse(BaseModel):
    status: str
    validated: bool

@app.post("/api/services/dunes_oauth_sync", response_model=SyncResponse)
async def dunes_oauth_sync(payload: SyncPayload):
    """
    Sync .MAML.ml data with OAuth2.0 and validate DUNES hash/signature.
    
    Args:
        payload (SyncPayload): OAuth token, MAML data, DUNES hash, and signature.
    
    Returns:
        SyncResponse: Sync and validation status.
    """
    try:
        # Validate JWT with RS256
        decoded = jwt.decode(payload.oauthToken, algorithms=["RS256"], verify=True)
        
        # Validate OAuth token with AWS Cognito
        headers = {"Authorization": f"Bearer {payload.oauthToken}"}
        auth_response = requests.post(
            "https://webxos.auth.us-east-1.amazoncognito.com/oauth2/token",
            headers=headers
        )
        auth_response.raise_for_status()
        
        # Verify DUNES hash and CRYSTALS-Dilithium signature
        hash_response = requests.post(
            "https://api.webxos.netlify.app/v1/dunes/verify_hash",
            json={"mamlData": payload.mamlData, "dunesHash": payload.dunesHash, "signature": payload.signature}
        )
        hash_response.raise_for_status()
        
        logger.info(f"DUNES OAuth sync completed for data hash: {payload.dunesHash}")
        return SyncResponse(status="success", validated=True)
    except Exception as e:
        logger.error(f"DUNES OAuth sync failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sync failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Deployment Instructions
# Path: webxos-vial-mcp/src/services/dunes_oauth_sync.py
# Run: pip install fastapi pydantic uvicorn requests pyjwt
# Start: uvicorn src.services.dunes_oauth_sync:app --host 0.0.0.0 --port 8000
