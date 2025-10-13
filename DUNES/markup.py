# markup.py: FastAPI routes for MARKUP Agent
# Purpose: Handles reverse Markdown (.mu) generation
# Instructions:
# 1. POST to /markup/generate_mu with Markdown content
# 2. Output is reversed content for error detection and receipts
from fastapi import APIRouter, HTTPException
from app.services.markup_agent import generate_mu_file

router = APIRouter(prefix="/markup", tags=["MARKUP Agent"])

@router.post("/generate_mu")
async def generate_mu(content: str):
    """
    Generate a .mu file with reversed Markdown content.
    Example: Input "Hello" -> Output "olleH" in reversed structure.
    """
    try:
        mu_content = generate_mu_file(content)
        return {"status": "success", "mu_content": mu_content}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"MARKUP generation error: {str(e)}")