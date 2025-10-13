# maml.py: FastAPI routes for MAML processing
# Purpose: Handles MAML file processing requests
# Instructions:
# 1. POST to /maml/process with MAML file content
# 2. Use example.maml.md for testing
from fastapi import APIRouter, HTTPException
from app.services.maml_processor import process_maml_file

router = APIRouter(prefix="/maml", tags=["MAML"])

@router.post("/process")
async def process_maml(file_content: str):
    """
    Process a MAML file and return parsed metadata and body.
    Example: Send content of example.maml.md via POST request.
    """
    try:
        result = process_maml_file(file_content)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"MAML processing error: {str(e)}")