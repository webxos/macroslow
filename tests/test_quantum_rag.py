import pytest
from backend.app.quantum_rag import QuantumRAG

@pytest.fixture
def quantum_rag():
    return QuantumRAG()

@pytest.mark.asyncio
async def test_quantum_rag_search(quantum_rag):
    results = quantum_rag.search("Test Query")
    assert isinstance(results, list)
    for result in results:
        assert "id" in result and "score" in result and "data" in result
    assert len(results) <= 5

@pytest.mark.asyncio
async def test_quantum_similarity(quantum_rag):
    query_vector = np.array([1.0, 0.0, 0.0])
    doc_vector = np.array([1.0, 0.0, 0.0])
    similarity = quantum_rag._quantum_similarity(query_vector, doc_vector)
    assert 0.9 <= similarity <= 1.0  # High similarity expected
