from fastapi import FastAPI
from server.security.agent_sandbox import router as agent_router

app = FastAPI(title="PROJECT DUNES 2048-AES MCP Server", version="1.0.7")

app.include_router(agent_router)

@app.get("/")
async def root():
    return {"message": "Welcome to PROJECT DUNES 2048-AES MCP Server"}