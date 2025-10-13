# 🐪 PROJECT DUNES 2048-AES: Comprehensive MAML Guide with .mu and .ms Integration

## Overview
Welcome to the **PROJECT DUNES 2048-AES SDK** guide for creating and processing **MAML (Markdown as Medium Language)** files, generating **.mu (Reverse Markdown)** files for error detection and auditability, and introducing **.ms (MAX Script)** files for mass prompt processing. This guide is designed for new users to understand and implement these formats within the DUNES ecosystem, with examples and instructions for integration with the hybrid MCP server.

### Purpose
- **.maml.md**: Structured, executable Markdown files with YAML front matter for secure data workflows.
- **.mu**: Reverse Markdown files for error detection, receipts, and ML training.
- **.ms**: MAX Script files for batch prompt submission to the MCP server, enabling mass AI interactions.

### Prerequisites
- **Python 3.8+** and dependencies from `requirements.txt` (e.g., FastAPI, PyTorch, SQLAlchemy).
- **MCP Server**: Running via `uvicorn app.main:app --reload` (see `README.md`).
- **Netlify/GitHub**: For deployment and version control (optional, see `README.md`).
- **Postman/cURL**: For testing API endpoints.

## Section 1: MAML (.maml.md) Files
MAML files are the core of the DUNES 2048-AES SDK, acting as secure, structured containers for workflows, datasets, and agent blueprints. They use YAML front matter for metadata and Markdown for content.

### MAML Structure
```yaml
---
schema: maml_v1
version: string
author: string
description: string
context: dict
input_schema: dict
output_schema: dict
---
# Markdown Content
Content body with executable code blocks, text, or data.
```

### Example 1: Basic MAML File
This example creates a simple MAML file with metadata and a Python code block.

```yaml
---
schema: maml_v1
version: 1.0
author: WebXOS Team
description: Basic MAML file with Python code
context:
  purpose: Data preprocessing
input_schema:
  data: list
output_schema:
  result: list
---
# Data Preprocessing Workflow
This MAML file defines a simple data preprocessing task.

## Code_Blocks
```python
def preprocess(data: list) -> list:
    return [x.upper() for x in data]
```
```

**Instructions**:
1. Save as `basic.maml.md`.
2. Send to `/maml/process` endpoint:
   ```bash
   curl -X POST http://localhost:8000/maml/process -H "Content-Type: text/plain" --data-binary @basic.maml.md
   ```
3. Response: Parsed metadata and body.

### Example 2: Quantum Workflow MAML
This example includes a Qiskit quantum circuit for quantum-resistant key generation.

```yaml
---
schema: maml_v1
version: 1.1
author: Quantum Team
description: Quantum key generation workflow
context:
  purpose: Cryptographic key generation
input_schema:
  qubits: int
output_schema:
  key: str
---
# Quantum Key Generation
This MAML file defines a quantum circuit for key generation.

## Code_Blocks
```python
from qiskit import QuantumCircuit
def generate_key(qubits: int) -> str:
    circuit = QuantumCircuit(qubits)
    circuit.h(range(qubits))
    circuit.measure_all()
    return "quantum_key"
```
```

**Instructions**:
1. Save as `quantum.maml.md`.
2. Send to `/maml/process` endpoint.
3. Extend `app/services/maml_processor.py` to execute Qiskit code in a sandbox.

## Section 2: Reverse Markdown (.mu) Files
The MARKUP Agent generates `.mu` files by reversing the structure and content of Markdown files (e.g., "Hello" → "olleH"). These files support error detection, auditability, and ML training.

### Example: Generating a .mu File
**Input (`basic.maml.md`)**:
```markdown
# Hello World
This is a test.
```

**POST to `/markup/generate_mu`**:
```bash
curl -X POST http://localhost:8000/markup/generate_mu -H "Content-Type: text/plain" --data-binary @basic.maml.md
```

**Output (`basic.mu`)**:
```markdown
.tset a si sihT
dlroW olleH #
```

**Instructions**:
1. Save input as `test.md`.
2. Send to `/markup/generate_mu` endpoint.
3. Use output for error detection or audit logging.

## Section 3: MAX Script (.ms) Files for Mass Prompts
**MAX Script (.ms)** is a new file format introduced as a custom upgrade for the DUNES 2048-AES SDK. It enables batch submission of prompts to the MCP server, ideal for large-scale AI interactions (e.g., testing multiple prompts with an AI model).

### MAX Script Structure
```yaml
---
schema: max_script_v1
version: string
author: string
description: string
prompts: list
---
# Prompt Definitions
List of prompts for batch processing.
```

### Example: MAX Script File
This example defines multiple prompts for AI processing.

```yaml
---
schema: max_script_v1
version: 1.0
author: WebXOS Team
description: Batch prompts for AI analysis
prompts:
  - id: prompt_1
    text: "Analyze the impact of quantum computing on cybersecurity."
  - id: prompt_2
    text: "Generate a summary of decentralized finance trends."
---
# Batch Prompt Processing
Send these prompts to the MCP server for processing.
```

**Instructions**:
1. Save as `batch_prompts.ms`.
2. Process using the custom `max_script_agent.py` (below).
3. Send to `/max_script/process` endpoint (requires new route).

### Custom Python Agent for MAX Script
Below is a new Python agent to process `.ms` files and integrate with the MCP server.

<xaiArtifact artifact_id="25cdd8f9-fb93-4935-9904-49f2a5b6dce9" artifact_version_id="7da65c41-1ba9-4ddd-abd4-cc76f26d0429" title="max_script_agent.py" contentType="text/python">

# max_script_agent.py: MAX Script processing for mass prompts
# Purpose: Processes .ms files for batch prompt submission to MCP server
# Instructions:
# 1. Add to app/services/max_script_agent.py
# 2. Add route in app/routes/max_script.py
# 3. POST to /max_script/process with .ms file content
import yaml
import re
import aiohttp
from typing import Dict, List

async def process_max_script(content: str) -> List[Dict[str, str]]:
    """
    Process a MAX Script (.ms) file and send prompts to MCP server.
    Args:
        content (str): Raw .ms file content
    Returns:
        list: Processed prompt responses
    Raises:
        ValueError: If YAML front matter or prompts are invalid
    """
    # Extract YAML front matter
    yaml_match = re.match(r'---\n(.*?)\n---\n(.*)', content, re.DOTALL)
    if not yaml_match:
        raise ValueError("Invalid MAX Script: Missing YAML front matter")
    
    metadata, body = yaml_match.groups()
    metadata = yaml.safe_load(metadata)
    
    # Validate schema and prompts
    if metadata.get("schema") != "max_script_v1":
        raise ValueError("Invalid schema: Must be max_script_v1")
    if "prompts" not in metadata:
        raise ValueError("MAX Script must include prompts in metadata")
    
    responses = []
    async with aiohttp.ClientSession() as session:
        for prompt in metadata["prompts"]:
            async with session.post(
                "http://localhost:8000/mcp/process",  # Replace with actual MCP endpoint
                json={"type": "custom", "content": prompt["text"], "token": "test_token"}
            ) as response:
                responses.append({
                    "prompt_id": prompt["id"],
                    "response": await response.json()
                })
    
    return responses