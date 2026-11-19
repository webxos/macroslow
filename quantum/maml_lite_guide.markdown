# 🐪 MAML-Lite Quick Start Guide

**Version:** 1.0.0  
**Publishing Entity:** WebXOS Research Group  
**Publication Date:** August 28, 2025  
**Copyright:** © 2025 Webxos. All Rights Reserved.

---

## Introduction

**MAML-Lite** is a simplified version of the **Markdown as Medium Language (MAML)** protocol, designed for developers new to Project Dunes and the Model Context Protocol (MCP). This guide provides a straightforward path to create and execute a basic `.maml.md` file using Python, with minimal dependencies and no quantum or OCaml requirements. It’s perfect for quick experimentation and integration with existing MCP servers.

---

## Setup

### Prerequisites
- Docker (for containerized deployment)
- Python 3.8+ (for local execution)
- A basic MCP server (e.g., a local JSON-RPC server)

### Step 1: Pull the MAML-Lite Docker Image
Run the following command to download the pre-configured MAML-Lite container:

```bash
docker pull webxos/maml-lite:latest
```

### Step 2: Start the MAML-Lite Gateway
Launch the container, exposing port 8000 for MCP communication:

```bash
docker run --gpus all -p 8000:8000 webxos/maml-lite
```

This starts a lightweight MAML gateway that supports Python code blocks and JSON-RPC communication with MCP servers.

---

## Creating Your First MAML File

Below is a simple `.maml.md` file that performs basic data processing with Pandas. Save it as `data_workflow.maml.md`.

```yaml
---
maml_version: "1.0.0"
id: "urn:uuid:123e4567-e89b-12d3-a456-426614174000"
type: "workflow"
origin: "agent://user-agent"
permissions:
  read: ["agent://*"]
  execute: ["gateway://local"]
created_at: 2025-08-28T23:00:00Z
---
## Intent
Process a CSV dataset and display the first five rows.

## Context
dataset: "input_data.csv"
output_format: "json"

## Code_Blocks
```python
import pandas as pd
df = pd.read_csv("input_data.csv")
result = df.head().to_json()
print(result)
```
```

### Running the MAML File
1. Ensure your MCP server is running and accessible at `localhost:8000`.
2. Place `input_data.csv` in the same directory as the MAML file.
3. Submit the file to the MAML-Lite gateway:

```bash
curl -X POST -H "Content-Type: text/markdown" --data-binary @data_workflow.maml.md http://localhost:8000/execute
```

The gateway will execute the Python code block and return the JSON output.

---

## Next Steps
- **Explore Templates**: Use The Architect’s `data-scientist-vial` template for pre-configured projects.
- **Add Validation**: Integrate Pydantic for data schema validation (see `maml_starter_notebook.ipynb`).
- **Scale Up**: Transition to full MAML with quantum and OCaml features for advanced use cases.

---

## Resources
- [MAML Language Guide](https://github.com/webxos/maml-language-guide)
- [Project Dunes Repository](https://github.com/webxos/project-dunes)
- [MCP Documentation](https://modelcontextprotocol.io)

**© 2025 Webxos. All Rights Reserved.**