from fastapi import WebSocket, Depends
from backend.app.annot8.main import app, manager
from backend.app.mcp.curator_server import CuratorServer
from backend.app.mcp.sentinel_server import SentinelServer
from sqlalchemy.orm import Session
from backend.app.database import get_db
import json

curator = CuratorServer()
sentinel = SentinelServer()

@app.websocket("/ws/enhanced/{client_id}")
async def enhanced_websocket(websocket: WebSocket, client_id: int, db: Session = Depends(get_db)):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            if await sentinel.validate_request({"token": data.get("token"), "user_id": "user1"}):
                validated_data = await curator.validate_data_schema(data)
                if validated_data["status"] == "valid":
                    annotation = {"text": data["text"], "x": data["x"], "y": data["y"], "user": "user1"}
                    await manager.broadcast_json(annotation)
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/annot8/enhanced")
async def get_enhanced_annotations(user=Depends(sentinel.validate_request), db: Session = Depends(get_db)):
    return {"annotations": [{"text": "enhanced", "x": 50, "y": 50}]}
