import pytest
from backend.app.mcp.validator_server import ValidatorServer, ValidationRequest

@pytest.mark.asyncio
async def test_validate_model():
    server = ValidatorServer()
    request = ValidationRequest(model_id="test_model", data_path="/data/test", threshold=0.5)
    await server.alchemist.train_model({"model_id": "test_model", "data_path": "/data/test", "epochs": 1})
    result = await server.validate_model(request)
    assert "status" in result
    assert "accuracy" in result

@pytest.mark.asyncio
async def test_invalid_model():
    server = ValidatorServer()
    request = ValidationRequest(model_id="nonexistent", data_path="/data/test", threshold=0.5)
    result = await server.validate_model(request)
    assert "error" in result
