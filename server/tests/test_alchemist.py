import pytest
from backend.app.mcp.alchemist_server import AlchemistServer, TrainingRequest

@pytest.mark.asyncio
async def test_train_model():
    server = AlchemistServer()
    request = TrainingRequest(model_id="test_model", data_path="/data/test", epochs=2)
    result = await server.train_model(request)
    assert result["model_id"] == "test_model"
    assert result["status"] == "trained"

@pytest.mark.asyncio
async def test_get_model_state():
    server = AlchemistServer()
    await server.train_model(TrainingRequest(model_id="test_model", data_path="/data/test", epochs=1))
    result = await server.get_model_state("test_model")
    assert "state" in result
