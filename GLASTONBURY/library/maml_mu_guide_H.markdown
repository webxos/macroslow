# MAML/MU Guide: H - HIPAA Compliance

## Overview
HIPAA compliance in MAML/MU ensures **INFINITY UI** protects medical data, critical for Nigerian healthcare applications using GLASTONBURY 2048’s encryption.

## Technical Details
- **MAML Role**: Specifies encrypted workflows in `cm_2048aes.py`.
- **MU Role**: Verifies compliance in `api_data_validator.py`.
- **Implementation**: Uses 2048-bit AES for all medical data, with OAuth for anonymous access.
- **Dependencies**: `pycryptodome`.

## Use Cases
- Secure patient data syncing from Nigerian hospital APIs.
- Protect IoT biometric data for SPACE HVAC environments.
- Ensure compliance in RAG-based medical datasets.

## Guidelines
- **Compliance**: Encrypt all data at rest and in transit (HIPAA).
- **Best Practices**: Implement audit logs for data access.
- **Code Standards**: Document encryption key management.

## Example
```python
encryptor = AES2048Encryptor()
data = torch.ones(1000).numpy().tobytes()
encrypted = encryptor.encrypt(data)
```