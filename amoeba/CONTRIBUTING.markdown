# Contributing to AMOEBA 2048AES SDK

## Description: Guidelines for contributing to the AMOEBA 2048AES SDK, including setup, coding standards, and pull request process.

## Getting Started
1. Fork the repository: [webxos/amoeba2048aes-sdk](https://github.com/webxos/amoeba2048aes-sdk)
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/amoeba2048aes-sdk.git
   ```
3. Install dependencies:
   ```bash
   bash install.sh
   ```

## Development Setup
- **Python**: 3.8+
- **Dependencies**: Install via `pip install -r requirements.txt`
- **Docker**: Required for containerized services
- **Dropbox**: Generate API tokens at [dropbox.com/developers](https://www.dropbox.com/developers)

## Coding Standards
- Follow PEP 8 for Python code.
- Use Pydantic for data validation.
- Include Gospel specifications (`*.mli`) for formal verification.
- Write tests using pytest in `test_amoeba2048.py`.
- Document MAML files with clear `Intent` and `Context` sections.

## Pull Request Process
1. Create a branch: `git checkout -b feature/your-feature`
2. Commit changes: `git commit -m "Add your feature"`
3. Push to your fork: `git push origin feature/your-feature`
4. Open a pull request against `main` with a clear description.
5. Ensure tests pass in GitHub Actions CI.

## Testing
Run tests locally:
```bash
pytest test_amoeba2048.py -v
```

## Issues
- Report bugs or feature requests at [Issues](https://github.com/webxos/amoeba2048aes-sdk/issues).
- Include reproduction steps and environment details.

## Community
Join the discussion on [GitHub Discussions](https://github.com/webxos/amoeba2048aes-sdk/discussions).

## License
Contributions are licensed under the [MIT License](LICENSE).