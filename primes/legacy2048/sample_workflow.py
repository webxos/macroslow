import asyncio
from src.legacy_2048.mcp_server import LegacyQuantumOrchestrator

# Team Instruction: Implement a sample workflow for prime sieving across four modes.
# Use MAML/MU for validation, inspired by Emeagwali’s orchestrated dataflow.
async def run_prime_sieve_workflow():
    orchestrator = LegacyQuantumOrchestrator()
    limit = 1000000
    node_signals = {"node_0": True, "node_1": True, "node_2": True, "node_3": True}
    maml_file = "workflows/prime_sieve.maml.md"
    primes = await orchestrator.execute_workflow(maml_file, limit, node_signals)
    print(f"Primes found: {primes[:10]}... Count: {len(primes)}")
    assert len(primes) == 78498, f"Expected 78498 primes, got {len(primes)}"

if __name__ == "__main__":
    asyncio.run(run_prime_sieve_workflow())