# 🐪 PROJECT DUNES 2048-AES: MXS Script (.mxs) Guide - Page 8: Community Contributions and Extending MXS Script

## Community Contributions and Extending MXS Script

This page provides a roadmap for community contributions and extending the functionality of **MXS Script (.mxs)** within the **PROJECT DUNES 2048-AES SDK**. As an open-source project, DUNES encourages developers, data scientists, and researchers to contribute to the MXS Script ecosystem, enhancing its capabilities for mass prompt processing, HTML/JavaScript integration, and interoperability with **MAML (.maml.md)** and **Reverse Markdown (.mu)**. This section outlines how new users can contribute, extend MXS Script with custom features, and collaborate with the WebXOS Research Group, aligning with the DUNES ecosystem’s minimalist hybrid Model Context Protocol (MCP) server architecture and its emphasis on secure, quantum-resistant, AI-orchestrated workflows.

### 1. Contributing to the MXS Script Ecosystem
The DUNES 2048-AES SDK is hosted on GitHub, and community contributions are essential for its growth. Contributions can include bug fixes, new features, documentation improvements, or example `.mxs` files.

#### How to Contribute
- **Fork the Repository**: Fork the DUNES repository at `github.com/webxos/dunes-2048-aes` (placeholder URL; check `README.md` for the official link).
- **Clone and Set Up**:
  ```bash
  git clone https://github.com/your-username/dunes-2048-aes.git
  cd dunes-2048-aes
  pip install -r requirements.txt
  ```
- **Create a Branch**:
  ```bash
  git checkout -b feature/your-feature-name
  ```
- **Make Changes**: Examples of contributions include:
  - Adding new prompt types to `mxs_script_agent.py`.
  - Enhancing JavaScript integration for UI applications like GalaxyCraft.
  - Improving `.mu` receipt generation for better auditability.
- **Test Changes**:
  - Write unit tests using `pytest`:
    ```python
    import pytest
    from app.services.mxs_script_agent import process_mxs_script
    @pytest.mark.asyncio
    async def test_new_prompt_type():
        content = """---
schema: mxs_script_v1
version: 1.0
author: Contributor
description: Test new prompt type
prompts:
  - id: new_prompt
    text: "Test new feature"
    type: custom
---
# Test
"""
        result = await process_mxs_script(content)
        assert result["status"] == "success"
    ```
  - Test with Postman against `/mxs_script/process`.
- **Submit a Pull Request**:
  - Push changes: `git push origin feature/your-feature-name`.
  - Create a pull request with a clear description of changes and tests.
- **Follow Guidelines**: Adhere to the DUNES contribution guidelines in `CONTRIBUTING.md`, including code style (PEP 8) and documentation standards.

**Instruction**: Review `CONTRIBUTING.md` and join the DUNES community on GitHub Discussions to propose ideas.

### 2. Extending MXS Script Functionality
MXS Script is designed to be extensible, allowing users to add custom features for specific use cases, such as new prompt types, advanced JavaScript triggers, or integration with external APIs.

#### Example: Adding a New Prompt Type
- **New Prompt Type**: Add support for “image_generation” prompts to generate images via an AI API (e.g., Stable Diffusion).
- **Update `mxs_script_agent.py`**:
  ```python
  async def process_mxs_script(content: str, execute_js: bool = False) -> Dict[str, any]:
      # ... existing code ...
      for prompt in metadata["prompts"]:
          if prompt.get("type") == "image_generation":
              async with session.post(
                  "https://api.stablediffusion.com/generate",  # Replace with actual API
                  json={"prompt": prompt["text"], "token": "api_token"}
              ) as response:
                  responses.append({
                      "prompt_id": prompt["id"],
                      "image_url": (await response.json())["url"]
                  })
          else:
              # Existing prompt processing
              async with session.post(
                  "http://localhost:8000/mcp/process",
                  json={"type": "custom", "content": prompt["text"], "token": "test_token"}
              ) as response:
                  responses.append({
                      "prompt_id": prompt["id"],
                      "response": await response.json()
                  })
      return {"status": "success", "prompt_responses": responses, "mu_receipt": mu_receipt}
  ```
- **Example MXS File**:
  ```yaml
  ---
  schema: mxs_script_v1
  version: 1.0
  author: Contributor
  description: Image generation prompts
  prompts:
    - id: image_001
      text: "Generate an image of a futuristic city."
      type: image_generation
      context:
        app: galaxycraft
        output: image
  javascript:
    - trigger: displayImage
      code: |
        document.getElementById('imageDisplay').innerHTML = `<img src="${response.data.image_url}" alt="Generated Image">`;
  ---
  # Image Generation
  Triggers image generation for GalaxyCraft.
  ```
- **Instructions**:
  1. Update `mxs_script_agent.py` with the new prompt type logic.
  2. Save the `.mxs` file as `image_gen.mxs`.
  3. Test with `/mxs_script/process?execute_js=true`.
  4. Submit as a pull request to the DUNES repository.

**Use Case**: Extends MXS Script for visual content generation in GalaxyCraft or other UI applications.

#### Example: Custom JavaScript Triggers
- **Custom Trigger**: Add a trigger for real-time data visualization in the 2048-AES SVG Diagram Tool.
- **Update MXS File**:
  ```yaml
  ---
  schema: mxs_script_v1
  version: 1.0
  author: Contributor
  description: Real-time SVG diagram update
  prompts:
    - id: diagram_001
      text: "Generate SVG data for a neural network diagram."
      context:
        app: svg_diagram_tool
        output: svg
  javascript:
    - trigger: visualizeDiagram
      code: |
        const svg = response.data.svg;
        const canvas = document.getElementById('svgCanvas');
        canvas.innerHTML = svg;
        canvas.querySelectorAll('path').forEach(p => p.classList.add('animate-pulse'));
  ---
  # SVG Diagram Visualization
  Triggers real-time SVG updates with animations.
  ```
- **Instructions**:
  1. Update the HTML interface (e.g., `dist/svg_diagram.html`) to handle the `visualizeDiagram` trigger.
  2. Test with `netlify deploy --prod`.
  3. Document the new trigger in `CONTRIBUTING.md`.

**Use Case**: Enhances the SVG Diagram Tool with animated visualizations.

### 3. Collaboration with the DUNES Community
The DUNES community encourages collaboration through GitHub, Discord, and other platforms.

- **Join Discussions**: Participate in GitHub Discussions to propose features, report bugs, or share `.mxs` examples.
- **Contribute Example Files**: Share `.mxs` files for use cases like GalaxyCraft content generation or GIBS Telescope captions:
  ```yaml
  ---
  schema: mxs_script_v1
  version: 1.0
  author: Community Contributor
  description: GIBS Telescope caption
  prompts:
    - id: caption_001
      text: "Generate a caption for a Mars satellite image."
      context:
        app: gibs_telescope
        source: nasa_api
  ---
  # GIBS Telescope Caption
  Shared by the community for NASA data visualization.
  ```
- **Hackathons and Events**: Join WebXOS-organized hackathons to develop MXS Script integrations for upcoming tools like Interplanetary Dropship Sim.
- **Documentation**: Improve the DUNES documentation by adding MXS Script tutorials or updating `README.md`.

**Instruction**: Submit example `.mxs` files via pull requests and join the DUNES Discord for real-time collaboration.

### 4. Best Practices for Extensions
- **Modularity**: Write reusable code in `mxs_script_agent.py` to support new prompt types or triggers.
- **Security**: Validate all user inputs and use quantum-resistant signatures for extensions:
  ```python
  from oqs import Signature
  sig = Signature('Dilithium2')
  if not sig.verify(content.encode(), signature, sig.public_key):
      raise ValueError("Invalid MXS signature")
  ```
- **Testing**: Include `pytest` tests for all new features:
  ```python
  @pytest.mark.asyncio
  async def test_image_prompt():
      content = """---
schema: mxs_script_v1
version: 1.0
author: Test
description: Image test
prompts:
  - id: image_001
    text: "Generate an image"
    type: image_generation
---
# Test
"""
      result = await process_mxs_script(content)
      assert "image_url" in result["prompt_responses"][0]
  ```
- **Documentation**: Update `docs/mxs_script.md` with new features and examples.

### Next Steps
Subsequent pages will cover:
- Future enhancements like ethical AI modules and natural language threat analysis.
- Final recommendations for MXS Script adoption.
- Summary of the MXS Script guide and community resources.

**Contact**: project_dunes@outlook.com  
**License**: MIT with attribution to WebXOS Research Group.