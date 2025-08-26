import re
import json
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class InputPayload(BaseModel):
    input: str

class SanitizedResponse(BaseModel):
    sanitizedInput: str
    status: str

def detect_malicious_intent(input_text: str) -> bool:
    """
    Detect prompt injection or malicious patterns.
    
    Args:
        input_text (str): Input to sanitize.
    
    Returns:
        bool: True if malicious, False otherwise.
    """
    blacklist_patterns = [
        r'malicious-command',
        r'prompt-injection-example',
        r'\bexec\b|\beval\b|\bsystem\b',
        r'[<>{}]'  # Basic HTML/script injection check
    ]
    return any(re.search(pattern, input_text, re.IGNORECASE) for pattern in blacklist_patterns)

@app.post("/api/services/sanitize", response_model=SanitizedResponse)
async def sanitize_input(payload: InputPayload):
    """
    Sanitize and validate input for agent communication.
    
    Args:
        payload (InputPayload): Input data to sanitize.
    
    Returns:
        SanitizedResponse: Sanitized input and status.
    """
    try:
        if detect_malicious_intent(payload.input):
            logger.warning(f"Malicious input detected: {payload.input}")
            raise HTTPException(status_code=400, detail="Malicious input detected")
        
        # Basic sanitization: remove sensitive patterns
        sanitized = re.sub(r'[^\w\s.,!?]', '', payload.input)
        logger.info(f"Input sanitized: {sanitized}")
        return SanitizedResponse(sanitizedInput=sanitized, status="success")
    except Exception as e:
        logger.error(f"Sanitization failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Sanitization failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Deployment Instructions
# Path: webxos-vial-mcp/src/services/communication_sanitizer.py
# Run: pip install fastapi pydantic uvicorn
# Start: uvicorn src.services.communication_sanitizer:app --host 0.0.0.0 --port 8000
