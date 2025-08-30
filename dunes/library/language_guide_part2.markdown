# The 2025 MAML/.mu Language Guide: Part 2 (D–I) 📚

Welcome to **Part 2** of the **2025 MAML/.mu Language Guide**! 🚀 This part covers commands from **D to I**, providing detailed syntax, usage, and examples for **MAML** and **.mu** in the **DUNES CORE SDK 2025**. Perfect for teams building agentic workflows and quantum-ready systems! 😄

### MAML Commands

#### `execute_workflow`
- **Description**: Executes a MAML workflow, running code blocks and processing sections.
- **Syntax**: `execute_workflow(maml_content: str) -> Dict`
- **Use**: Runs Python code in `Code_Blocks` sections and returns results.
- **Example**:
  ```python
  from dunes_workflow import DunesWorkflow
  workflow = DunesWorkflow()
  maml_content = open("examples/ml_pipeline.maml").read()
  result = await workflow.execute_workflow(maml_content)
  print(result)
  ```
  **Output**:
  ```json
  {
    "status": "Executed",
    "output": "import torch\nmodel = torch.nn.Linear(10, 1)"
  }
  ```

### .mu Commands

#### `invert_structure`
- **Description**: Inverts the structure of a MAML file to create a `.mu` file.
- **Syntax**: `invert_structure(parsed_maml: Dict) -> str`
- **Use**: Reverses section order and YAML keys for `.mu` generation.
- **Example**:
  ```python
  from dunes_markup import MarkupAgent
  markup = MarkupAgent("sqlite:///dunes_logs.db")
  maml_content = open("examples/ml_pipeline.maml").read()
  parsed = markup._parse_markdown(maml_content)
  mu_content = markup._reverse_to_markup(parsed)
  print(mu_content)
  ```
  **Output**:
  ```markdown
  ---
  eltit: eniPeline LM
  noisrev_lmam: 0.0.1
  di: ...
  ---
  ## skcolB_edoC
  ```python
  import torch
  model = torch.nn.Linear(10, 1)
  ```
  ## evitcebjO
  krowten laruen a niarT
  ```

---

## 📚 Tips for Teams
- Use `execute_workflow` to automate ML pipelines or data tasks.
- `invert_structure` is internal but critical for `.mu` receipt integrity.