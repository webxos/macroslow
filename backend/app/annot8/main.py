from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends, Request
from fastapi.templating import Jinja2Templates
from .auth import get_current_user, create_access_token, User, Annotation
from .websockets import ConnectionManager
from sqlalchemy.orm import Session
from backend.app.database import get_db
import datetime

app = FastAPI()
templates = Jinja2Templates(directory="../frontend/templates")
manager = ConnectionManager()

@app.get("/")
async def get_index_page(request: Request):
    return templates.TemplateResponse("index.html.j2", {"request": request})

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int, db: Session = Depends(get_db)):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            user = await get_current_user(data.get("token"), db)
            annotation = Annotation(text=data["text"], x_percent=data["x"], y_percent=data["y"], user_id=user.id)
            db.add(annotation)
            db.commit()
            await manager.broadcast_json({"user": user.name, "text": data["text"], "x": data["x"], "y": data["y"]})
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/api/annotations")
async def get_annotations(user=Depends(get_current_user), db: Session = Depends(get_db)):
    annotations = db.query(Annotation).filter(Annotation.user_id == user.id).all()
    return [{"id": a.id, "text": a.text, "x_percent": a.x_percent, "y_percent": a.y_percent, "created_at": a.created_at} for a in annotations]

@app.get("/api/annotations/export")
async def export_annotations(user=Depends(get_current_user), db: Session = Depends(get_db)):
    annotations = db.query(Annotation).filter(Annotation.user_id == user.id).all()
    export_data = [{"text": a.text, "x_percent": a.x_percent, "y_percent": a.y_percent, "created_at": a.created_at.isoformat()} for a in annotations]
    return {"format": "json", "data": export_data}
