import pytest
from fastapi.testclient import TestClient
from backend.app.annot8.main import app
from backend.app.mcp.sentinel_server import SentinelServer

client = TestClient(app)
sentinel = SentinelServer()

def test_export_endpoint():
    response = client.get("/api/annot8/export?format=csv", headers={"Authorization": "Bearer valid_token"})
    assert response.status_code == 200
    assert "data" in response.json()
    assert response.json()["format"] == "csv"

@pytest.mark.asyncio
async def test_unauthorized_export():
    response = client.get("/api/annot8/export?format=csv", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Unauthorized"
