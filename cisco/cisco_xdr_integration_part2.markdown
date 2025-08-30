# Cisco XDR Integration Guide: Part 2 - Telemetry Integration and Processing 📡

Welcome to **Part 2**! 🚀 This part integrates Cisco XDR telemetry (endpoint, firewall, etc.) into the DUNES CORE SDK’s MCP server, processing data with MAML workflows. Let’s connect and analyze! 😄

## 🌟 Overview
- **Goal**: Fetch and process Cisco XDR telemetry using MAML and .mu.
- **Tools**: Cisco XDR API, DUNES MAML parser, SQLAlchemy for logging.
- **Use Cases**: Threat correlation, incident prioritization, and recursive training.

## 📋 Steps

### 1. Create MAML Workflow
Create `cisco/xdr_maml_workflow.maml` to define telemetry processing.

<xaiArtifact artifact_id="b97094e2-f0ec-4b2e-80db-4b13bcc9d883" artifact_version_id="e5d80bfd-a6c0-4ce0-9842-6d55f9c55b81" title="cisco/xdr_maml_workflow.maml" contentType="text/markdown">
```markdown
---
title: Cisco XDR Telemetry Workflow
maml_version: 1.0.0
id: urn:uuid:987fcdeb-1234-5678-9abc-def123456789
---
## Objective
Fetch and process Cisco XDR endpoint telemetry
## Code_Blocks
```python
import requests
from cisco_xdr_config import CiscoXDRConfig

config = CiscoXDRConfig.load_from_env()
token = config.get_access_token()
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(f"{config.api_base_url}/telemetry/endpoint", headers=headers)
print(response.json())
```
```