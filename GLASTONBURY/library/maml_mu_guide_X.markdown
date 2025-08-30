# MAML/MU Guide: X - eXporting Data

## Overview
Data exporting in MAML/MU generates MAML-formatted files for **INFINITY UI**, using GLASTONBURY 2048’s `maml_formatter.py` for API/IoT data outputs.

## Technical Details
- **MAML Role**: Defines export workflows in `infinity_server.py`.
- **MU Role**: Validates exported files in `infinity_workflow_validation.mu.md`.
- **Implementation**: Uses YAML for MAML-compliant Markdown formatting.
- **Dependencies**: `pyyaml`.

## Use Cases
- Export medical records for Nigerian clinics.
- Generate IoT data reports for SPACE HVAC systems.
- Create RAG datasets in MAML format.

## Guidelines
- **Compliance**: Ensure exported data is encrypted (HIPAA-compliant).
- **Best Practices**: Use unique IDs for export files.
- **Code Standards**: Document export formats for consistency.

## Example
```python
formatter = MAMLFormatter()
data = {"patient_id": "12345", "vitals": [120, 95, 1.5]}
output_file = formatter.format(data)
```