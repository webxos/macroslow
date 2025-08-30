# The 2025 MAML/.mu Language Guide: Part 4 (P–T) 📚

Welcome to **Part 4** of the **2025 MAML/.mu Language Guide**! 🚀 This part covers commands from **P to T**, helping you parse, secure, and train with **MAML** and **.mu**. Let’s keep building! 😄

### MAML Commands

#### `parse_maml`
- **Description**: Parses a MAML file into a structured dictionary.
- **Syntax**: `parse_maml(content: str) -> Dict`
- **Use**: Converts MAML into a format for processing or validation.
- **Example**:
  ```python
  from dunes_maml_parser import DunesMAMLParser
  parser = DunesMAMLParser()
  maml_content = open("examples/ml_pipeline.maml").read()
  parsed = parser.parse_maml(maml_content)
  print(parsed)
  ```
  **Output**:
  ```json
  {
    "front_matter": {"title": "ML Pipeline", "maml_version": "1.0.0", "id": "..."},
    "sections": {
      "Objective": ["Train a neural network"],
      "Code_Blocks": ["import torch", "model = torch.nn.Linear(10, 1)"]
    }
  }
  ```

#### `secure_workflow`
- **Description**: Secures a MAML workflow with authentication and guardrails.
- **Syntax**: `secure_workflow(maml_content: str, token: str) -> List[str]`
- **Use**: Ensures workflows are safe from prompt injection and unauthorized access.
- **Example**:
  ```python
  from dunes_security import DunesSecurity
  security = DunesSecurity()
  token = await security.auth.get_access_token("user@example.com", "password")
  errors = await security.secure_workflow("---\ntitle: Test\n---\n## Objective\nTest", token)
  print(errors)
  ```
  **Output**: `[]` (if token and content are valid)

### .mu Commands

#### `train_recursive`
- **Description**: Trains an agentic recursion network on MAML/.mu pairs.
- **Syntax**: `train_recursive(limit: int = 1000)`
- **Use**: Enhances error detection and model performance using mirrored data.
- **Example**:
  ```python
  from dunes_arn import DunesARN
  arn = DunesARN("sqlite:///dunes_logs.db")
  await arn.train_recursive(limit=10)
  print("Training completed!")
  ```

---

## 📚 Tips for Teams
- Use `parse_maml` to build custom processors or validators.
- Combine `secure_workflow` with OAuth for secure APIs.
- Run `train_recursive` to improve ML models with `.mu` receipts.