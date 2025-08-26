import pytest
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.mcp_tools import MCPTools, MAMLToolInput, MAMLToolOutput

client = TestClient(app)
mcp_tools = MCPTools()

@pytest.mark.asyncio
async def test_maml_execute():
    sample_maml = """
    ---
    maml_version: "0.1.0"
    id: "urn:uuid:test-123"
    type: "workflow"
    origin: "agent://test-agent"
    permissions:
      execute: ["test-user"]
    ---
    ## Intent
    Test execution.
    ## Code_Blocks
    ```python
    print("Test")
    ```
    """
    result = await mcp_tools.maml_execute(MAMLToolInput(maml_content=sample_maml, user_id="test-user"))
    assert result.status == "success"
    assert "python_output" in result.result["outputs"]

@pytest.mark.asyncio
async def test_maml_create():
    sample_maml = """
    ---
    maml_version: "0.1.0"
    id: "urn:uuid:test-456"
    type: "workflow"
    origin: "agent://test-agent"
    ---
    ## Intent
    Test creation.
    """
    result = await mcp_tools.maml_create(MAMLToolInput(maml_content=sample_maml, user_id="test-user"))
    assert result.status == "success"
    assert "id" in result.result

@pytest.mark.asyncio
async def test_maml_validate():
    sample_maml = """
    ---
    maml_version: "0.1.0"
    id: "urn:uuid:test-789"
    type: "workflow"
    origin: "agent://test-agent"
    ---
    ## Intent
    Test validation.
    ## Code_Blocks
    ```python
    print("Test")
    ```
    """
    result = await mcp_tools.maml_validate(MAMLToolInput(maml_content=sample_maml, user_id="test-user"))
    assert result.status == "success"

@pytest.mark.asyncio
async def test_maml_search():
    result = await mcp_tools.maml_search(MAMLToolInput(maml_content="Test", user_id="test-user"))
    assert result.status == "success"
    assert isinstance(result.result, list)
