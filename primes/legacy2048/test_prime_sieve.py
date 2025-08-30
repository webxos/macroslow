import pytest
import torch
from src.legacy_2048.mcp_server import LegacyQuantumOrchestrator
from src.legacy_2048.maml_validator import MAMLValidator
from src.legacy_2048.mu_validator import MUValidator

# Team Instruction: Implement tests for prime sieving workflow.
# Ensure CUDA-accelerated sieving and MAML/MU validation, inspired by Emeagwali’s rigorous testing.
@pytest.fixture
def orchestrator():
    return LegacyQuantumOrchestrator()

def test_maml_validation():
    validator = MAMLValidator()
    result = validator.validate("workflows/prime_sieve.maml.md")
    assert result["status"] == "valid"

def test_mu_validation():
    validator = MUValidator()
    result = validator.validate("workflows/prime_sieve_validation.mu.md")
    assert result["status"] == "valid"

@pytest.mark.asyncio
async def test_prime_sieve(orchestrator):
    limit = 1000000
    node_signals = {"node_0": True, "node_1": True, "node_2": True, "node_3": True}
    primes = await orchestrator.execute_workflow("workflows/prime_sieve.maml.md", limit, node_signals)
    assert len(primes) == 78498  # Expected prime count for 10^6
    assert primes[:5] == [2, 3, 5, 7, 11]

if __name__ == "__main__":
    pytest.main(["-v"])