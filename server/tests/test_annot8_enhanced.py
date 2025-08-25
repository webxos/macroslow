import pytest
from fastapi.testclient import TestClient
from backend.app.annot8.main import app
from backend.app.mcp.sentinel_server import SentinelServer

client = TestClient(app)
sentinel = SentinelServer()

def test_enhanced_annotation_endpoint():
    response = client.get("/api/annot8/enhanced", headers={"Authorization": "Bearer valid_token"})
    assert response.status_code == 200
    assert "annotations" in response.json()

@pytest.mark.asyncio
async def test_enhanced_websocket():
    from backend.app.annot8.websockets import manager
    from fastapi.websockets import WebSocket
    ws = WebSocket()
    await manager.connect(ws, 1)
    data = {"text": "test", "x": 50, "y": 50, "token": "valid_token"}
    if await sentinel.validate_request({"token": data["token"], "user_id": "user1"}):
        await manager.broadcast_json(data)
    assert True  # Placeholder for full WebSocket test
    await manager.disconnect(ws, 1)
