# 🐪 PROJECT DUNES 2048-AES: MXS Script (.mxs) Guide - Page 3: Technical Implementation of MXS Script

## Technical Implementation of MXS Script

This page provides a detailed guide to implementing **MXS Script (.mxs)** within the **PROJECT DUNES 2048-AES SDK**, focusing on its structure, processing logic, and integration with the hybrid **Model Context Protocol (MCP)** server, **MAML (.maml.md)**, and **Reverse Markdown (.mu)**. MXS Script is designed for mass prompt processing, enabling scalable AI interactions, and supports advanced scripting capabilities, including HTML/JavaScript integration inspired by Autodesk’s MAXScript. This section is tailored for new users, offering step-by-step instructions to set up MXS Script processing, including a custom Python agent, FastAPI routes, and bidirectional communication for web-based interfaces.

### MXS Script Structure
MXS Script (.mxs) files are structured to facilitate batch AI prompt processing, using YAML front matter for metadata and Markdown for documentation, mirroring the design of MAML files. The structure supports both prompt definitions and optional JavaScript code for web-based interactivity.

#### MXS File Format
```yaml
---
schema: mxs_script_v1
version: string
author: string
description: string
prompts:
  - id: string
    text: string
    context: dict (optional)
javascript: (optional)
  - trigger: string
    code: string
---
# Prompt Definitions
Markdown content documenting the prompts or JavaScript functionality.
```

- **schema**: Must be `mxs_script_v1` to ensure compatibility.
- **version**: Indicates the MXS Script version (e.g., `1.0`).
- **author**: Identifies the creator (e.g., `WebXOS Team`).
- **description**: Describes the file’s purpose (e.g., `Batch AI prompts`).
- **prompts**: A list of prompt objects, each with an `id`, `text`, and optional `context`.
- **javascript**: Optional list of JavaScript snippets with triggers for HTML integration.
- **Markdown Body**: Provides documentation or instructions for processing.

#### Example MXS File
```yaml
---
schema: mxs_script_v1
version: 1.0
author: WebXOS Team
description: Batch AI prompts with JavaScript UI trigger
prompts:
  - id: prompt_1
    text: "Analyze quantum computing trends."
    context:
      domain: cybersecurity
  - id: prompt_2
    text: "Summarize Web3 advancements."
javascript:
  - trigger: updateUI
    code: |
      document.getElementById('output').innerHTML = response.data.result;
---
# Batch Prompt Processing
Send prompts to the MCP server and update UI with JavaScript.
```

**Instructions**:
1. Save as `batch_prompts.mxs`.
2. Process using the custom agent described below.
3. Use the `/mxs_script/process` endpoint to handle prompts and JavaScript triggers.

### Processing Logic: Custom MXS Script Agent
To process MXS Script files, a custom Python agent (`mxs_script_agent.py`) is required to parse the file, validate its structure, and route prompts to the MCP server. The agent also supports generating `.mu` receipts for auditability and executing JavaScript for HTML interfaces.

#### Custom Agent Code
Below is the implementation of `mxs_script_agent.py`, designed to process `.mxs` files and integrate with the MCP server.

<xaiArtifact artifact_id="08c27dee-7e15-4b72-91be-8063c33f5c8a" artifact_version_id="dc4d1a3e-ff6a-4b65-9c15-346ad445f08f" title="mxs_script_agent.py" contentType="text/python">

# mxs_script_agent.py: MXS Script processing for mass prompts
# Purpose: Parses .mxs files, processes prompts, and supports HTML/JavaScript integration
# Instructions:
# 1. Save in app/services/mxs_script_agent.py
# 2. Add route in app/routes/mxs_script.py
# 3. POST to /mxs_script/process with .mxs file content
# 4. Extend for JavaScript execution in HTML interfaces
import yaml
import re
import aiohttp
from typing import Dict, List
from app.services.markup_agent import generate_mu_file

async def process_mxs_script(content: str, execute_js: bool = False) -> Dict[str, any]:
    """
    Process an MXS Script (.mxs) file, send prompts to MCP server, and generate .mu receipt.
    Args:
        content (str): Raw .mxs file content
        execute_js (bool): Whether to process JavaScript triggers
    Returns:
        dict: Processed prompt responses and .mu receipt
    Raises:
        ValueError: If YAML front matter or prompts are invalid
    """
    # Extract YAML front matter
    yaml_match = re.match(r'---\n(.*?)\n---\n(.*)', content, re.DOTALL)
    if not yaml_match:
        raise ValueError("Invalid MXS Script: Missing YAML front matter")
    
    metadata, body = yaml_match.groups()
    metadata = yaml.safe_load(metadata)
    
    # Validate schema and prompts
    if metadata.get("schema") != "mxs_script_v1":
        raise ValueError("Invalid schema: Must be mxs_script_v1")
    if "prompts" not in metadata:
        raise ValueError("MXS Script must include prompts in metadata")
    
    # Process prompts
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
    
    # Generate .mu receipt for auditability
    mu_receipt = generate_mu_file(content)
    
    # Handle JavaScript triggers (placeholder for HTML integration)
    js_results = []
    if execute_js and "javascript" in metadata:
        for js in metadata["javascript"]:
            js_results.append({
                "trigger": js["trigger"],
                "code": js["code"],
                "status": "Pending execution in HTML context"
            })
    
    return {
        "status": "success",
        "prompt_responses": responses,
        "mu_receipt": mu_receipt,
        "javascript_results": js_results
    }