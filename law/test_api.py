# test_api.py
# Description: Unit tests for the Lawmakers Suite 2048-AES FastAPI server. Verifies core API endpoints (root, query, resources) to ensure functionality and reliability. Uses pytest for automated testing, suitable for integration into CI/CD pipelines for legal research platform development.

import pytest
from fastapi.testclient import TestClient
from server.main import app

client = TestClient(app)

def test_root():
    """
    Test the root endpoint to ensure the API is running.
    """
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Lawmakers Suite 2048-AES API is running"}

def test_query_endpoint():
    """
    Test the query endpoint with a sample legal query.
    """
    response = client.post("/query", json={"text": "Sample legal query"})
    assert response.status_code == 200
    assert "encrypted_query" in response.json()
    assert response.json()["status"] == "success"

def test_resources_endpoint():
    """
    Test the resources endpoint to ensure it returns available data sources.
    """
    response = client.get("/resources")
    assert response.status_code == 200
    assert "resources" in response.json()
    assert isinstance(response.json()["resources"], list)

if __name__ == "__main__":
    pytest.main(["-v", "test_api.py"])