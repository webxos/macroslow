from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from backend.app.maml_parser import MAMLParser
from backend.app.maml_executor import MAMLExecutor
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
db = MongoDBClient()

@app.post("/api/maml/upload")
async def upload_maml(file: UploadFile = File(...), user=Depends(get_current_user)):
    content = await file.read()
    maml_data = parser.parse(content.decode())
    db.save_maml(maml_data["id"], maml_data)
    return {"message": "MAML file uploaded", "id": maml_data["id"]}

@app.get("/api/maml/{maml_id}")
async def get_maml(maml_id: str, user=Depends(get_current_user)):
    maml_data = db.get_maml(maml_id)
    if maml_data:
        return maml_data
    return {"error": "MAML not found"}, 404

@app.post("/api/maml/execute/{maml_id}")
async def execute_maml(maml_id: str, user=Depends(get_current_user)):
    maml_data = db.get_maml(maml_id)
    if maml_data:
        result = await executor.execute(maml_data)
        db.update_maml_history(maml_id, {"timestamp": "2025-08-25T18:50:00Z", "action": "EXECUTE", "status": "Success"})
        return result
    return {"error": "MAML not found"}, 404

@app.websocket("/ws/maml")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            maml_data = parser.parse(data)
            result = await executor.execute(maml_data)
            await websocket.send_text(str(result))
    except WebSocketDisconnect:
        pass
