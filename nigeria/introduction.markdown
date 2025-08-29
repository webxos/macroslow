# Decentralized Unified Nigerian Empowerment Stack

**Version**: 1.0.0  
**Inspired by**: Philip Emeagwali's Connection Machine  
**Mission**: To empower Nigerian developers with a quantum-ready, high-performance SDK and server framework for Web3, AI, and advanced computational fields, fostering global collaboration and innovation.

## Overview

The **DUNES Project** is an open-source initiative that reimagines Philip Emeagwali’s Connection Machine as a modern, decentralized computational fabric. The project provides a Python-based SDK (`dunes-sdk`) and a Model Context Protocol (MCP) server, leveraging NVIDIA CUDA for accelerated computing, Qiskit for quantum simulation, and OCaml/Ortac for formal verification. The SDK enables developers to build applications that perform parallel computations across four logical nodes (the "Quadrilinear Core"), simulating a high-throughput system suitable for AI/ML, data analysis, and quantum workflows.

## Features

- **Quadrilinear Engine**: A parallel processing framework that distributes computations across four nodes, inspired by Emeagwali’s 65,536-processor design.
- **Quantum Integration**: Qiskit-based quantum circuit simulation with CUDA acceleration for hybrid quantum-classical workflows.
- **MCP Server**: A Model Context Protocol server for standardized communication with AI systems, supporting real-time video processing and quantum-enhanced retrieval-augmented generation (RAG).
- **Formal Verification**: OCaml and Ortac integration for mathematically verified workflows, ensuring correctness in high-stakes applications.
- **Humanitarian Focus**: Designed to empower Nigerian developers through accessible tools, educational resources, and global hackathon opportunities.

## Prerequisites

- **Hardware**:
  - NVIDIA GPU with 8GB+ VRAM (Recommended: RTX 4090 with 24GB VRAM)
  - 16GB+ System RAM
  - 4-core+ CPU
  - 100GB+ SSD Storage
- **Software**:
  - Python 3.9+
  - NVIDIA CUDA Toolkit 12.2
  - Docker and NVIDIA Container Toolkit
  - OCaml 4.14+ and Ortac for formal verification
  - Qiskit for quantum computing
- **Dependencies**:
  ```bash
  pip install torch numpy sqlalchemy modelcontextprotocol qiskit qiskit-aer
  sudo apt install nvidia-driver-535 nvidia-cuda-toolkit
  opam install ortac
  ```

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/webxos/dunes-sdk.git
   cd dunes-sdk
   ```

2. **Set Up Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # venv\Scripts\activate  # Windows
   pip install -e .
   ```

3. **Install NVIDIA Container Toolkit**:
   ```bash
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker
   ```

4. **Build and Run with Docker**:
   ```bash
   docker build --build-arg CUDA_VERSION=12.2 -t dunes-mcp-server .
   docker run --gpus all -p 8000:8000 -p 9090:9090 -p 3000:3000 dunes-mcp-server
   ```

## Usage

1. **Run the MCP Server**:
   ```bash
   python src/dunes_sdk/protocols/mcp_server.py
   ```
   The server will be available at `localhost:8000` for MCP clients to connect and execute parallel computations or quantum simulations.

2. **Execute a Sample Quantum Workflow**:
   Use the provided `verifiable_simulator.maml.md` file to run a verified quantum circuit simulation:
   ```bash
   dunes-sdk run src/dunes_sdk/quantum/verifiable_simulator.maml.md
   ```

## Contributing

We welcome contributions from the global developer community:
- **Fork and Contribute**: Add features, fix bugs, or improve documentation at `https://github.com/webxos/dunes-sdk`.
- **Host a Node**: Run one of the four Quadrilinear Core nodes to support the network.
- **Mentor**: Guide Nigerian developers in using the SDK and MCP server.
- **Spread the Word**: Share the project through blogs, social media, or hackathons.

## Roadmap

- **Q1 2025**: Solidify quadrilinear engine and MCP server compatibility.
- **Q2 2025**: Integrate OCaml/Ortac for formal verification of workflows.
- **Q3 2025**: Add quantum simulation capabilities with Qiskit and CUDA acceleration.
- **Q4 2025**: Launch a global "Build for Nigeria" hackathon to showcase the SDK.

## License

MIT License. © 2025 Webxos Technologies.
