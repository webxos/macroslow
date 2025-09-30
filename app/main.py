from fastapi import FastAPI
import torch

app = FastAPI(title="PROJECT DUNES 2048-AES API")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }