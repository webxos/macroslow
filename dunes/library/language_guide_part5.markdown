# The 2025 MAML/.mu Language Guide: Part 5 (U–Z) 📚

Welcome to **Part 5** of the **2025 MAML/.mu Language Guide**! 🚀 This final part covers commands from **U to Z**, completing the glossary for **MAML** and **.mu**. Let’s wrap up with validation and visualization! 😄

### MAML Commands

#### `validate_maml`
- **Description**: Validates a MAML file’s structure and syntax.
- **Syntax**: `validate_maml(content: str) -> List[str]`
- **Use**: Checks for valid YAML front matter and section structure.
- **Example**:
  ```python
  from dunes_maml import DunesMAML
  maml = DunesMAML()
  maml_content = open("examples/ml_pipeline.maml").read()
  errors = maml.validate_maml(maml_content)
  print("Valid:", len(errors) == 0)
  ```
  **Output**: `Valid: True`

#### `visualize_workflow`
- **Description**: Generates a 3D graph of a MAML workflow or `.mu` transformation.
- **Syntax**: `visualize_workflow(graph_data: Dict)`
- **Use**: Visualizes data flows for debugging and analysis.
- **Example**:
  ```python
  from dunes_visualizer import DunesVisualizer
  visualizer = DunesVisualizer()
  graph_data = {
      "nodes": [{"id": "maml", "label": "MAML Input"}, {"id": "mu", "label": "MARKUP Receipt"}],
      "edges": [{"from": "maml", "to": "mu"}]
  }
  visualizer.render_3d_graph(graph_data)
  ```
  **Output**: A file `dunes_graph.html` with a 3D visualization.

### .mu Commands

#### `validate_receipt`
- **Description**: Validates a `.mu` receipt against its MAML source.
- **Syntax**: `validate_receipt(maml_content: str, receipt_content: str) -> List[str]`
- **Use**: Ensures the `.mu` receipt correctly mirrors the MAML file.
- **Example**:
  ```python
  from dunes_receipts import DunesReceipts
  receipts = DunesReceipts("sqlite:///dunes_logs.db")
  maml_content = open("examples/ml_pipeline.maml").read()
  receipt_content = open("examples/ml_pipeline.mu").read()
  errors = await receipts.validate_receipt(maml_content, receipt_content)
  print("Valid:", len(errors) == 0)
  ```
  **Output**: `Valid: True`

---

## 📚 Tips for Teams
- Use `validate_maml` and `validate_receipt` to ensure workflow integrity.
- Create visualizations with `visualize_workflow` to debug complex pipelines.