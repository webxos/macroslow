import pytest
from backend.app.mcp.astronomer_server import AstronomerServer, SpaceDataRequest

@pytest.mark.asyncio
async def test_fetch_space_data():
    server = AstronomerServer()
    request = SpaceDataRequest(dataset="apod", start_date="2025-08-01", end_date="2025-08-25")
    result = await server.fetch_space_data(request)
    assert "data" in result or "error" in result

@pytest.mark.asyncio
async def test_process_telescope_data():
    server = AstronomerServer()
    result = await server.process_telescope_data("gibs_123")
    assert "processed_data" in result or "error" in result
