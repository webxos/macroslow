import pytest
from fastapi.testclient import TestClient
from backend.app.annot8.main import app
from backend.app.database import SessionLocal
from backend.app.annot8.auth import User, Annotation

client = TestClient(app)

def test_annotation_endpoint():
    response = client.get("/api/annotations", headers={"Authorization": "Bearer test_token"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)

@pytest.mark.asyncio
async def test_websocket_broadcast():
    from backend.app.annot8.websockets import manager
    from fastapi.websockets import WebSocket
    ws = WebSocket()
    await manager.connect(ws, 1)
    await manager.broadcast_json({"user": "test", "text": "test", "x": 50, "y": 50})
    await manager.disconnect(ws, 1)
    # Note: Full WebSocket testing requires a mock server

def test_export_annotations():
    db = SessionLocal()
    user = User(provider_id="test", email="test@example.com", name="Test User")
    db.add(user)
    db.commit()
    annot = Annotation(text="test", x_percent=50, y_percent=50, user_id=user.id)
    db.add(annot)
    db.commit()
    response = client.get("/api/annotations/export", headers={"Authorization": "Bearer test_token"})
    assert response.status_code == 200
    assert "data" in response.json()
    db.delete(annot)
    db.delete(user)
    db.commit()
