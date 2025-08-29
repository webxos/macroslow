# AMOEBA 2048AES Dropbox Addon

## Overview
The **AMOEBA 2048AES Dropbox Addon** integrates the Dropbox API with the AMOEBA 2048AES SDK, enabling users to store and retrieve MAML files, quantum circuits, and task results in Dropbox. This addon supports quadralinear workflows (Compute, Quantum, Security, Orchestration) with quantum-safe signatures, MCP/MAML compliance, and Project Dunes orchestration.

## Features
- **Dropbox File Management**: Upload and download MAML files and task results securely.
- **Quantum-Safe Security**: Sign and verify files using Dilithium2 cryptography.
- **Quadralinear Execution**: Execute tasks from Dropbox-hosted MAML files across CHIMERA heads.
- **MCP/MAML Integration**: Seamless interaction with Model Context Protocol servers.
- **Monitoring**: Prometheus metrics for upload/download latency and signature verification.

## Getting Started

### Prerequisites
- AMOEBA 2048AES SDK (see `README.md`)
- Dropbox account with API access (generate tokens at https://www.dropbox.com/developers)
- NVIDIA GPU with 8GB+ VRAM
- Docker and NVIDIA Container Toolkit
- Python 3.8+, Qiskit, PyTorch, Dropbox SDK, Cryptography

### Installation
1. Install Dropbox SDK:
   ```bash
   pip install dropbox==11.36.2
   ```
2. Configure Dropbox credentials in `dropbox_config.yml`:
   ```yaml
   dropbox:
     access_token: "your_dropbox_access_token"
     app_key: "your_dropbox_app_key"
     app_secret: "your_dropbox_app_secret"
   ```
3. Build and run the Docker container:
   ```bash
   docker build -f Dockerfile.dropbox -t amoeba2048_dropbox .
   docker run --gpus all -p 8000:8000 -p 9090:9090 amoeba2048_dropbox
   ```

### Usage
1. Create a MAML file for a quadralinear task:
   ```bash
   cp dropbox_maml_handler.maml.md my_workflow.maml.md
   ```
2. Run a Dropbox-integrated workflow:
   ```bash
   python3 dropbox_integration.py
   ```
3. Monitor performance via Prometheus:
   ```bash
   curl http://localhost:9090
   ```

## Project Structure
- `dropbox_integration.py`: Core Dropbox API integration module.
- `dropbox_maml_handler.maml.md`: MAML template for Dropbox workflows.
- `dropbox_config.yml`: Configuration for Dropbox and security settings.
- `Dockerfile.dropbox`: Dockerfile for containerized deployment.
- `README.dropbox.md`: This file.

## Integration with Project Dunes
- Deploy the addon as part of the DUNES API Gateway using Vercel:
  ```bash
  vercel deploy --prod
  ```
- Submit MAML files to the DUNES orchestrator:
  ```bash
  curl -X POST http://your-vercel-url/execute-task -d '{"maml_file": "workflow.maml.md", "task_id": "sample_task"}'
  ```

## Next Steps
- Customize `dropbox_maml_handler.maml.md` for specific workflows.
- Integrate with DUNES API Gateway for distributed execution.
- Explore Prometheus metrics for performance optimization.

## License
© 2025 Webxos. All Rights Reserved.

[](https://github.com/dropbox/dropbox-sdk-python)