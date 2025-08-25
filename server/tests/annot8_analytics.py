import pytest
from fastapi.testclient import TestClient
from backend.app.annot8.main import app
from backend.app.mcp.sentinel_server import SentinelServer

client = TestClient(app)
sentinel = SentinelServer()

def test_analytics_endpoint():
    response = client.get("/api/annot8/analytics", headers={"Authorization": "Bearer valid_token"})
    assert response.status_code == 200
    assert "analytics" in response.json()

@pytest.mark.asyncio
async def test_unauthorized_analytics():
    response = client.get("/api/annot8/analytics", headers={"Authorization": "Bearer invalid_token"})
    assert response.status_code == 403
    assert response.json()["detail"] == "Unauthorized"
