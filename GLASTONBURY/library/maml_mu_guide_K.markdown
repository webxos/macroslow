# MAML/MU Guide: K - Key Management

## Overview
Key management in MAML/MU ensures secure encryption for **INFINITY UI**, using GLASTONBURY 2048’s 2048-bit AES keys for API and IoT data.

## Technical Details
- **MAML Role**: Specifies key usage in `cm_2048aes.py`.
- **MU Role**: Verifies key integrity in `api_data_validator.py`.
- **Implementation**: Generates session-specific keys with `pycryptodome`.
- **Dependencies**: `pycryptodome`.

## Use Cases
- Secure medical data syncing in Nigerian hospitals.
- Protect IoT sensor data for SPACE HVAC systems.
- Ensure key security for RAG datasets.

## Guidelines
- **Compliance**: Store keys securely (HIPAA/GDPR-compliant).
- **Best Practices**: Rotate keys daily for enhanced security.
- **Code Standards**: Document key generation processes.

## Example
```python
encryptor = AES2048Encryptor()
key = encryptor.generate_key()
```