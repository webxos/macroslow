# main.py: Core FastAPI application for PROJECT DUNES 2048-AES SDK
# Purpose: Initializes the MCP server and routes requests to MAML and MARKUP endpoints
# Instructions:
# 1. Ensure requirements.txt is installed
# 2. Configure .env with necessary credentials
# 3. Run with: uvicorn app.main:app --reload
from fastapi import FastAPI
from dotenv import load_dotenv
from app.routes import maml, markup
import os

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="PROJECT DUNES 2048-AES MCP Server",
    description="Hybrid MCP server for MAML processing and MARKUP Agent"
)

# Include routers for MAML and MARKUP endpoints
app.include_router(maml.router, prefix="/maml")
app.include_router(markup.router, prefix="/markup")

@app.get("/")
async def root():
    return {
        "message": "Welcome to PROJECT DUNES 2048-AES MCP Server",
        "docs": "/docs for API documentation"
    }