# QFN Installation Guide

## Introduction

This guide details how to install and deploy the **Quantum Fortran Network (QFN)** for production use. It covers setting up Fortran servers, the Python SDK, PostgreSQL, and security configurations to ensure a robust, secure system.

## Prerequisites

- **OS**: Ubuntu 22.04+ (or compatible Linux distribution)
- **Fortran Compiler**: `gfortran` or `nvfortran` (for CUDA support)
- **Python**: 3.11+
- **Database**: PostgreSQL 15+
- **Hardware**: Optional NVIDIA GPU for CUDA acceleration
- **Dependencies**:
  - Fortran: `fpm`, `fsocket`, `quadtensor_lib`
  - Python: See `sdk/python/requirements.txt`
  - Tools: `git`, `docker`, `docker-compose`

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/your-org/qfn-system.git
cd qfn-system
```

### 2. Install Fortran Dependencies
```bash
# Install fpm
curl -sL https://github.com/fortran-lang/fpm/releases/latest/download/fpm-install.sh | bash
# Install dependencies
fpm install
```

### 3. Install Python Dependencies
```bash
pip install -r sdk/python/requirements.txt
# Optional: Install PyTorch with CUDA
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### 4. Set Up PostgreSQL
```bash
sudo apt update
sudo apt install postgresql
sudo -u postgres createdb qfn_state
sudo -u postgres psql -c "CREATE USER qfn_user WITH PASSWORD 'secure_pass';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE qfn_state TO qfn_user;"
```

### 5. Configure Environment
Create `sdk/python/.env`:
```env
DATABASE_URL=postgresql://qfn_user:secure_pass@localhost:5432/qfn_state
AES_MASTER_KEY=your-2048-bit-key
AES_KEY_SEGMENT_1=segment1
AES_KEY_SEGMENT_2=segment2
AES_KEY_SEGMENT_3=segment3
AES_KEY_SEGMENT_4=segment4
SERVER_1_HOST=localhost:50051
SERVER_2_HOST=localhost:50052
SERVER_3_HOST=localhost:50053
SERVER_4_HOST=localhost:50054
MAX_REQUESTS_PER_SECOND=100
```

### 6. Build and Run Servers
```bash
fpm run --target qfn_server_1
# Repeat for servers 2, 3, 4 in separate terminals
```

### 7. Deploy with Docker (Optional)
```bash
docker-compose up -d
```

### 8. Test the SDK
```bash
cd sdk/python
python qfn_sdk.py
```

## Security Configurations

- **Prompt Shielding**:
  - Enable input validation in `qfn_sdk.py` to reject malformed tensors.
  - Example: Add checks for tensor size < 1MB.
- **Guardrails**:
  - Set `MAX_REQUESTS_PER_SECOND` in `.env` to limit DoS attacks.
  - Use RBAC for PostgreSQL: Restrict `qfn_user` to `qfn_state` database.
- **Encryption**:
  - Generate a 2048-bit AES key using a secure tool (e.g., `openssl`).
  - Store keys in a secrets manager (e.g., HashiCorp Vault).
- **Network Security**:
  - Deploy behind a firewall, exposing only ports 50051–50054.
  - Use TLS for gRPC (configure in `qfn.proto` channels).

## Troubleshooting

- **Server Fails to Start**: Check if ports 50051–50054 are in use (`netstat -tuln`).
- **Database Connection Error**: Verify `DATABASE_URL` and PostgreSQL credentials.
- **Dependency Issues**: Ensure `fpm` and `pip` installations completed successfully.

# Embedded Guidance: Save this file in `docs/`. Follow each step carefully, ensuring environment variables are set securely. Use a secrets manager for `AES_MASTER_KEY`. Test the system with the `example_tensor_transform.ipynb` notebook after setup.