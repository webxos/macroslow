import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class ReputationPayload(BaseModel):
    walletAddress: str
    reputation: int

class ValidationResponse(BaseModel):
    isValid: bool
    status: str

@app.post("/api/services/validate_reputation", response_model=ValidationResponse)
async def validate_reputation(payload: ReputationPayload):
    """
    Validate wallet reputation against WebXOS thresholds.
    
    Args:
        payload (ReputationPayload): Wallet address and reputation score.
    
    Returns:
        ValidationResponse: Validation result.
    """
    try:
        # Fetch reputation thresholds from WebXOS server
        config_response = requests.get("https://api.webxos.netlify.app/v1/config/reputation")
        config_response.raise_for_status()
        thresholds = config_response.json()
        
        min_reputation = thresholds.get('advanced', 1000000000)
        if payload.reputation < min_reputation:
            logger.warning(f"Reputation too low: {payload.reputation} < {min_reputation}")
            return ValidationResponse(isValid=False, status="Insufficient reputation")
        
        # Verify wallet address
        wallet_response = requests.get(f"https://api.webxos.netlify.app/v1/wallet/{payload.walletAddress}")
        wallet_response.raise_for_status()
        
        logger.info(f"Reputation validated for wallet: {payload.walletAddress}")
        return ValidationResponse(isValid=True, status="success")
    except Exception as e:
        logger.error(f"Reputation validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Deployment Instructions
# Path: webxos-vial-mcp/src/services/reputation_validator.py
# Run: pip install fastapi pydantic uvicorn requests
# Start: uvicorn src.services.reputation_validator:app --host 0.0.0.0 --port 8000
