# AMOEBA 2048AES SDK Security Policy

## Description: Security policy for reporting vulnerabilities in the AMOEBA 2048AES SDK.

## Supported Versions
| Version | Supported          |
|---------|--------------------|
| 1.0.0   | ✅                 |

## Reporting a Vulnerability
- Email vulnerabilities to: security@webxos.org
- Include:
  - Detailed description of the issue
  - Steps to reproduce
  - Potential impact
- Expect a response within 48 hours.
- Patches will be released promptly for critical issues.

## Security Features
- Quantum-safe cryptography (Dilithium2) for MAML signatures.
- Secure Dropbox API integration with token-based authentication.
- Formal verification via Ortac for CHIMERA heads.

## Best Practices
- Keep `.env` files secure and never commit them.
- Validate MAML files with `maml_validator.py`.
- Monitor security metrics via Prometheus.