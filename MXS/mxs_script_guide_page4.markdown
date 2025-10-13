# 🐪 PROJECT DUNES 2048-AES: MXS Script (.mxs) Guide - Page 4: Practical Examples of MXS Script

## Practical Examples of MXS Script

This page provides practical examples of **MXS Script (.mxs)** usage within the **PROJECT DUNES 2048-AES SDK**, demonstrating its versatility for AI-driven workflows and HTML/JavaScript integration. These examples illustrate how to create `.mxs` files for mass prompt processing, integrate with the Model Context Protocol (MCP) server, and leverage **MAML (.maml.md)** and **Reverse Markdown (.mu)** for secure, auditable AI interactions. Each example includes instructions for implementation, tailored for new users, and aligns with the DUNES ecosystem’s minimalist hybrid MCP server architecture.

### Example 1: Batch AI Prompt Processing
This `.mxs` file defines multiple prompts for analyzing technology trends, processed via the MCP server.

```yaml
---
schema: mxs_script_v1
version: 1.0
author: WebXOS Team
description: Batch prompts for AI-driven trend analysis
prompts:
  - id: prompt_1
    text: "Analyze the impact of quantum computing on cybersecurity."
    context:
      domain: cybersecurity
      priority: high
  - id: prompt_2
    text: "Summarize recent advancements in decentralized finance."
    context:
      domain: fintech
      priority: medium
  - id: prompt_3
    text: "Predict AI trends for 2026."
    context:
      domain: artificial intelligence
      priority: low
---
# Batch Prompt Processing
Send these prompts to the MCP server for analysis by an AI model.
```

**Instructions**:
1. Save as `trend_analysis.mxs`.
2. Send to the `/mxs_script/process` endpoint:
   ```bash
   curl -X POST http://localhost:8000/mxs_script/process -H "Content-Type: text/plain" --data-binary @trend_analysis.mxs
   ```
3. The `mxs_script_agent.py` processes each prompt and generates a `.mu` receipt for auditability.
4. Response includes prompt IDs and AI-generated outputs, sanitized by `SecurityFilter`.

**Use Case**: Ideal for researchers analyzing trends across multiple domains, with `.mu` receipts ensuring data integrity.

### Example 2: Interactive UI Prompting with HTML/JavaScript
This `.mxs` file integrates with a web-based interface (e.g., GalaxyCraft) to trigger prompts and update the UI dynamically, inspired by MAXScript’s HTML integration.

```yaml
---
schema: mxs_script_v1
version: 1.0
author: WebXOS Team
description: Interactive prompts for GalaxyCraft UI
prompts:
  - id: galaxy_prompt
    text: "Generate a description for a fictional galaxy."
    context:
      app: GalaxyCraft
      output: text
javascript:
  - trigger: updateGalaxy
    code: |
      document.getElementById('galaxyOutput').innerHTML = response.data.result;
      document.getElementById('galaxyOutput').classList.add('text-green-600');
---
# Interactive Galaxy Prompt
Trigger this prompt from a web UI to update the GalaxyCraft interface.
```

**HTML Interface** (save as `dist/galaxycraft.html`):
```html
<!DOCTYPE html>
<html>
<head>
    <title>GalaxyCraft MXS Prompt</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-white p-6">
    <div class="max-w-lg mx-auto">
        <h1 class="text-3xl font-bold mb-4">GalaxyCraft Prompt Interface</h1>
        <button id="triggerGalaxy" class="bg-blue-600 text-white p-3 rounded hover:bg-blue-700">Generate Galaxy</button>
        <div id="galaxyOutput" class="mt-4 p-4 bg-gray-800 rounded"></div>
    </div>
    <script>
        document.getElementById('triggerGalaxy').addEventListener('click', async () => {
            const mxsContent = `---
schema: mxs_script_v1
version: 1.0
author: WebXOS Team
description: Interactive prompts for GalaxyCraft UI
prompts:
  - id: galaxy_prompt
    text: "Generate a description for a fictional galaxy."
    context:
      app: GalaxyCraft
      output: text
javascript:
  - trigger: updateGalaxy
    code: |
      document.getElementById('galaxyOutput').innerHTML = response.data.result;
      document.getElementById('galaxyOutput').classList.add('text-green-600');
---
# Interactive Galaxy Prompt
Trigger this prompt from a web UI.
`;
            try {
                const response = await fetch('http://localhost:8000/mxs_script/process?execute_js=true', {
                    method: 'POST',
                    headers: { 'Content-Type': 'text/plain' },
                    body: mxsContent
                });
                const result = await response.json();
                result.javascript_results.forEach(js => {
                    if (js.trigger === 'updateGalaxy') {
                        eval(js.code); // Use sandboxed execution in production
                    }
                });
            } catch (error) {
                document.getElementById('galaxyOutput').innerHTML = `Error: ${error.message}`;
            }
        });
    </script>
</body>
</html>
```

**Instructions**:
1. Save the `.mxs` content as `galaxycraft.mxs` or embed in the HTML script.
2. Save the HTML as `dist/galaxycraft.html` and deploy via `netlify deploy --prod`.
3. Click the button to send the prompt and update the UI with the AI response.
4. Ensure `mxs_script_agent.py` is configured to handle `execute_js=true`.

**Use Case**: Enhances interactive applications like GalaxyCraft, where AI-generated content dynamically updates the UI.

### Example 3: Hybrid MAML and MXS Workflow
This `.mxs` file triggers a MAML workflow for quantum key generation, combining prompt processing with executable code.

```yaml
---
schema: mxs_script_v1
version: 1.0
author: WebXOS Team
description: Trigger MAML quantum workflow
prompts:
  - id: quantum_prompt
    text: "Generate a quantum key and describe its security."
    context:
      maml_file: quantum_workflow.maml.md
---
# Quantum Workflow Trigger
This MXS file triggers a MAML workflow for quantum key generation.
```

**Related MAML File** (save as `quantum_workflow.maml.md`):
```yaml
---
schema: maml_v1
version: 1.0
author: WebXOS Team
description: Quantum key generation workflow
context:
  purpose: Cryptographic key generation
input_schema:
  qubits: int
output_schema:
  key: str
---
# Quantum Key Generation
## Code_Blocks
```python
from qiskit import QuantumCircuit
def generate_key(qubits: int) -> str:
    circuit = QuantumCircuit(qubits)
    circuit.h(range(qubits))
    circuit.measure_all()
    return "quantum_key_123"
```
```

**Instructions**:
1. Save both files in the project directory.
2. Update `mxs_script_agent.py` to handle MAML file references:
   ```python
   async def process_mxs_script(content: str, execute_js: bool = False) -> Dict[str, any]:
       # ... existing code ...
       for prompt in metadata["prompts"]:
           if "maml_file" in prompt.get("context", {}):
               with open(prompt["context"]["maml_file"], "r") as f:
                   maml_content = f.read()
               from app.services.maml_processor import process_maml_file
               maml_result = process_maml_file(maml_content)
               responses.append({
                   "prompt_id": prompt["id"],
                   "maml_result": maml_result
               })
           # ... continue with HTTP requests for other prompts ...
   ```
3. Send `quantum_trigger.mxs` to `/mxs_script/process`.
4. Generate a `.mu` receipt for the combined workflow.

**Use Case**: Combines MXS prompt orchestration with MAML’s executable workflows for hybrid AI-quantum tasks.

### Best Practices
- **Structure**: Ensure `.mxs` files include `schema: mxs_script_v1` and valid prompt definitions.
- **Security**: Use OAuth2.0 (AWS Cognito) and `SecurityFilter` to sanitize responses.
- **JavaScript**: Avoid `eval` in production; use a sandboxed JavaScript engine.
- **Auditability**: Store `.mu` receipts in SQLAlchemy for compliance.
- **Testing**: Use Postman to test endpoints with sample `.mxs` files.

### Next Steps
Subsequent pages will cover:
- Advanced MXS Script features (e.g., blockchain audit trails).
- Best practices for secure, scalable MXS workflows.
- Integration with DUNES’ future UI developments (e.g., GalaxyCraft, SVG Diagram Tool).

**Contact**: project_dunes@outlook.com  
**License**: MIT with attribution to WebXOS Research Group.