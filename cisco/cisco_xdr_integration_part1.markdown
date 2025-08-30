# Cisco XDR Integration Guide: Part 1 - Setup and Configuration ⚙️

Welcome to **Part 1** of the **2025 Cisco XDR Integration Guide** for the **DUNES CORE SDK**! 🎉 This guide helps production-level teams integrate **Cisco XDR** with the **Model Context Protocol (MCP)** server, using **MAML** and **.mu** for secure, scalable workflows. In this part, we set up the SDK, configure Cisco XDR API access, and create initial MAML workflows. Perfect for beginners and enterprise teams! 😄

## 🌟 Overview
- **Goal**: Configure the DUNES CORE SDK and Cisco XDR API for telemetry integration.
- **Tools**: Python 3.10+, Cisco XDR API, FastAPI, SQLAlchemy, MAML/.mu.
- **Use Cases**: Threat detection, incident response, and agentic automation.

## 📋 Steps

### 1. Install Dependencies
Create `cisco/requirements.txt` to install required libraries.

<xaiArtifact artifact_id="a2c85e82-6b8e-4fa0-8c58-c83704dbd8e6" artifact_version_id="6449689f-5e1d-4229-b1cf-a1e9c622ea30" title="cisco/requirements.txt" contentType="text/plain">
# Dependencies for Cisco XDR + DUNES CORE SDK integration
torch>=2.0.0  # ML and quantum processing
sqlalchemy>=2.0.0  # Database logging
fastapi>=0.100.0  # MCP server API
uvicorn>=0.20.0  # API server
pyyaml>=6.0  # MAML parsing
plotly>=5.10.0  # Visualization
pydantic>=2.0.0  # Data validation
requests>=2.28.0  # Cisco XDR API calls
python-jose[cryptography]>=3.3.0  # OAuth2.0
psutil>=5.9.0  # System monitoring
qiskit>=0.44.0  # Quantum validation
qiskit-aer>=0.12.0  # Quantum simulation