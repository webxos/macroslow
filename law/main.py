# main.py
# Description: Core FastAPI server for the Lawmakers Suite 2048-AES. Handles API requests, query processing, and data encryption using AES-256. Integrates with MCP for managing data flows from LLMs and legal databases. Provides endpoints for legal research queries and secure data transmission. Designed for scalability and secure operation in legal environments.

from fastapi import FastAPI
from dotenv import load_dotenv
import os
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from fastapi.responses import JSONResponse

app = FastAPI(title="Lawmakers Suite 2048-AES API")

# Load environment variables
load_dotenv()

# AES-256 encryption function
def encrypt_data(data: str, key: bytes) -> bytes:
    iv = os.urandom(16)  # Generate random initialization vector
    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    padded_data = data + " " * (16 - len(data) % 16)  # Pad data to block size
    encrypted = encryptor.update(padded_data.encode()) + encryptor.finalize()
    return iv + encrypted  # Prepend IV for decryption

# Root endpoint for API health check
@app.get("/")
async def root():
    return {"message": "Lawmakers Suite 2048-AES API is running"}

# Query endpoint for processing legal research queries
@app.post("/query")
async def process_query(query: dict):
    key = os.getenv("AES_KEY").encode()
    if not key:
        return JSONResponse(status_code=500, content={"error": "AES key not configured"})
    try:
        encrypted_query = encrypt_data(query.get("text", ""), key)
        return {"encrypted_query": encrypted_query.hex(), "status": "success"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# MCP resource discovery endpoint (placeholder for integration with legal databases)
@app.get("/resources")
async def list_resources():
    # Example: Connect to legal databases or LLMs (to be extended)
    return {"resources": ["Bloomberg Law", "Hugging Face LLM", "CourtListener"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)