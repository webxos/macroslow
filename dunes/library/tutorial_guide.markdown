# DUNES CORE SDK 2025 Tutorial Guide for Beginners 🎉

Welcome to the **DUNES CORE SDK 2025 Tutorial Guide**! 🚀 If you're new to **Project Dunes** or just starting with machine learning, software development, or agentic systems, this guide is your friendly introduction to building a **Model Context Protocol (MCP)** server using the **DUNES CORE SDK**. This lightweight, open-source toolkit lets you create powerful workflows with **MAML (Markdown Agentic Markup Language)** 📝 and generate self-checking **MARKUP (.mu)** receipts 🧾, optimized for 2025’s world of AI, quantum computing, and recursive learning. No experience? No problem! We’ll walk you through every step with clear examples and plenty of emojis to keep it fun! 😄

This tutorial covers:
1. **Setting Up the SDK** ⚙️: Installing and configuring the DUNES CORE SDK.
2. **Writing Your First MAML Workflow** 📜: Creating a simple MAML file.
3. **Generating .mu Receipts** 🧾: Using the SDK to create mirrored receipts.
4. **Running the MCP Server** 🌐: Launching the API and testing endpoints.
5. **Visualizing Workflows** 📊: Creating 3D graphs for debugging.
6. **Next Steps** 🚀: Customizing and contributing to Project Dunes.

By the end, you’ll have a working MCP server and the skills to start building your own AI-driven projects! Let’s dive in! 🌟

---

## 🌟 Step 1: Setting Up the DUNES CORE SDK

Let’s get the SDK up and running on your computer! 🖥️

### Prerequisites
- **Python 3.10+** 🐍: Download from [python.org](https://www.python.org).
- **pip**: Python’s package manager (comes with Python).
- **Git**: For cloning the repository (install from [git-scm.com](https://git-scm.com)).
- **Optional**: Docker 🐳 for containerized deployment.

### Installation
1. **Clone the Repository** 📥:
   Open a terminal and run:
   ```bash
   git clone https://github.com/your-org/dunes_core_sdk.git
   cd dunes_core_sdk
   ```

2. **Install Dependencies** 📦:
   Create a `requirements.txt` file with the following content:

   <xaiArtifact artifact_id="921a031c-3439-4ce8-8b87-7b8d5bfa359b" artifact_version_id="e2df2a54-ac07-4165-b5f9-5bbf4006eda2" title="requirements.txt" contentType="text/plain">
   torch
   sqlalchemy
   fastapi
   uvicorn
   pyyaml
   plotly
   pydantic
   requests
   qiskit
   qiskit-aer
   python-jose[cryptography]
   psutil