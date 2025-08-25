import pytest
from backend.app.mcp.rube_optimizer import RubeOptimizer

@pytest.mark.asyncio
async def test_optimize_tool_execution():
    server = RubeOptimizer()
    result = await server.optimize_tool_execution("slack_send_message", {"channel": "#test", "message": "test"})
    assert "status" in result

@pytest.mark.asyncio
async def test_invalid_tool():
    server = RubeOptimizer()
    result = await server.optimize_tool_execution("invalid_tool", {})
    assert "error" in result
