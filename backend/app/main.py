from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.app.maml_parser import MAMLParser
from backend.app.maml_executor import MAMLExecutor
from backend.app.mcp_tools import MCPTools
from backend.app.database import MongoDBClient
from backend.app.auth import get_current_user

app = FastAPI(title="MAML Gateway API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
parser = MAMLParser()
executor = MAMLExecutor()
mcp_tools = MCPTools()
db = MongoDBClient()

@app.post("/api/maml/upload")
async def upload_maml(file: UploadFile = File(...), user=Depends(get_current_user)):
    content = await file.read()
    return await mcp_tools.maml_create(MAMLToolInput(maml_content=content.decode(), user_id=user["username"]))

@app.get("/api/maml/{maml_id}")
async def get_maml(maml_id: str, user=Depends(get_current_user)):
    maml_data = db.get_maml(maml_id)
    if maml_data and user["username"] in maml_data["metadata"]["permissions"]["read"]:
        return maml_data
    return {"error": "MAML not found or unauthorized"}, 403

@app.post("/api/maml/execute/{maml_id}")
async def execute_maml(maml_id: str, user=Depends(get_current_user)):
    maml_data = db.get_maml(maml_id)
    if maml_data and user["username"] in maml_data["metadata"]["permissions"]["execute"]:
        content = str(maml_data)
        return await mcp_tools.maml_execute(MAMLToolInput(maml_content=content, user_id=user["username"]))
    return {"error": "MAML not found or unauthorized"}, 403

@app.post("/api/maml/validate")
async def validate_maml(file: UploadFile = File(...), user=Depends(get_current_user)):
    content = await file.read()
    return await mcp_tools.maml_validate(MAMLToolInput(maml_content=content.decode(), user_id=user["username"]))

@app.get("/api/maml/search")
async def search_maml(query: str, user=Depends(get_current_user)):
    return await mcp_tools.maml_search(MAMLToolInput(maml_content=query, user_id=user["username"]))

@app.websocket("/ws/maml")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            result = await mcp_tools.maml_execute(MAMLToolInput(maml_content=data, user_id="websocket_user"))
            await websocket.send_text(str(result))
    except WebSocketDisconnect:
        pass
