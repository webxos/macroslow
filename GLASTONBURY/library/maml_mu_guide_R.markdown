# MAML/MU Guide: R - Retrieval-Augmented Generation (RAG)

## Overview
RAG in MAML/MU enables **INFINITY UI** to use API/IoT data as a temporary knowledge base, avoiding client-side training, integrated with GLASTONBURY 2048.

## Technical Details
- **MAML Role**: Defines RAG workflows in `infinity_server.py`.
- **MU Role**: Validates RAG outputs in `infinity_workflow_validation.mu.md`.
- **Implementation**: Caches API data for real-time processing with CUDA.
- **Dependencies**: PyTorch, FastAPI.

## Use Cases
- Generate medical datasets for Nigerian healthcare AI.
- Process IoT data for SPACE HVAC analytics.
- Create RAG-based legal document datasets.

## Guidelines
- **Compliance**: Ensure data minimization for GDPR compliance.
- **Best Practices**: Clear RAG cache after each session.
- **Code Standards**: Document RAG data sources.

## Example
```python
config = Config()
config.data_cache = await fetch_api_data("https://api.example.com/data")
```