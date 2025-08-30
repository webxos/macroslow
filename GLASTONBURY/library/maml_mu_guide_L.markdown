# MAML/MU Guide: L - Legacy System Compatibility

## Overview
Legacy system compatibility in MAML/MU ensures **INFINITY UI** integrates with outdated infrastructure, critical for Nigerian healthcare using GLASTONBURY 2048.

## Technical Details
- **MAML Role**: Defines legacy workflows in `c64_512aes.py`.
- **MU Role**: Validates outputs for legacy systems in `infinity_workflow_validation.mu.md`.
- **Implementation**: Uses C64 emulation for compatibility, integrated with CUDA.
- **Dependencies**: `libsdl2-dev`.

## Use Cases
- Sync medical records with legacy Nigerian hospital systems.
- Process IoT data on outdated hardware for SPACE HVAC.
- Generate compatible RAG datasets for older AI models.

## Guidelines
- **Compliance**: Ensure data encryption for legacy systems (HIPAA).
- **Best Practices**: Test on emulated C64 environments.
- **Code Standards**: Document compatibility layers.

## Example
```python
c64 = C64_512AES()
data = torch.ones(1000, device="cuda")
result = c64.process(data)
```