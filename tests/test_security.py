import pytest
from backend.app.maml_security import MAMLSecurity
from backend.app.maml_parser import MAMLParser

@pytest.fixture
def security():
    return MAMLSecurity()

@pytest.mark.asyncio
async def test_sandbox_execution(security):
    code = 'print("Secure Test")\nopen("/etc/passwd", "r")'  # Attempt insecure operation
    result = security.sandbox_execute(code, "python")
    assert "Error" in result and "Permission denied" in result

@pytest.mark.asyncio
async def test_quantum_signature(security):
    sample_maml = {"metadata": {"id": "test-123"}, "sections": {"Intent": "Test"}}
    enhanced_maml = security.enhance_signature(sample_maml)
    assert "quantum_signature" in enhanced_maml["metadata"]
    assert enhanced_maml["metadata"]["quantum_signature"].startswith("Q-SIGN-")

@pytest.mark.asyncio
async def test_parser_validation():
    parser = MAMLParser()
    invalid_maml = "Invalid content"
    with pytest.raises(ValueError):
        parser.parse(invalid_maml)
