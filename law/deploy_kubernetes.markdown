# Kubernetes Deployment Guide for Lawmakers Suite 2048-AES
## Description
This guide outlines the deployment of the Lawmakers Suite 2048-AES, a full-service agentic quantum API hub gateway for legal research, on a Kubernetes cluster. It supports Angular for a dynamic frontend, Python for backend and data science, and advanced cryptographic modes (AES-256, AES-512, quantum-parallel AES-2048). The platform integrates legal databases (Westlaw, LexisNexis, CourtListener), LLMs, and a secure networking hub for OBS video feeds. Designed for law schools in the 2025 fall season, it ensures scalability, security, and support for forensic, archaeological, and biological data science.

## Prerequisites
- **Kubernetes Cluster**: AWS EKS, Google GKE, or Azure AKS with `kubectl` configured.
- **Helm**: v3.12+ installed.
- **Docker Images**:
  - Backend: `your-repo/lawmakers-api:latest` (Python/FastAPI).
  - Frontend: `your-repo/lawmakers-angular:latest` (Angular).
- **Hardware**: NVIDIA GPUs for CUDA support (PyTorch simulations); optional quantum hardware for Qiskit.
- **Credentials**: `.env` file with API keys (`WESTLAW_API_KEY`, `LEXISNEXIS_API_KEY`, `COURTLISTENER_API_KEY`, `OPENAI_API_KEY`, `HUGGINGFACE_API_KEY`) and cryptographic keys (`AES_KEY`, `QUANTUM_KEY`).
- **Software**: Node.js 18, Python 3.9+, Qiskit, PyTorch, SQLAlchemy.

## Deployment Steps
1. **Clone Repository**:
   ```bash
   git clone https://github.com/your-repo/lawmakers-suite-2048-aes.git
   cd lawmakers-suite-2048-aes
   ```

2. **Build and Push Docker Images**:
   - Backend:
     ```bash
     docker build -t your-repo/lawmakers-api:latest -f Dockerfile .
     docker push your-repo/lawmakers-api:latest
     ```
   - Angular Frontend:
     ```bash
     docker build -t your-repo/lawmakers-angular:latest -f Dockerfile.angular .
     docker push your-repo/lawmakers-angular:latest
     ```

3. **Configure Cryptography**:
   - Generate AES-256/512 keys and quantum-parallel AES-2048 keys:
     ```bash
     python scripts/generate_keys.py
     ```
   - Update `.env` with generated keys (see `updated_env` below).

4. **Install Helm Charts**:
   - Backend:
     ```bash
     helm install lawmakers-api ./helm/lawmakers-suite \
       --set image.repository=your-repo/lawmakers-api \
       --set image.tag=latest \
       --set env.AES_KEY=$(cat .env | grep AES_KEY | cut -d'=' -f2) \
       --set env.QUANTUM_KEY=$(cat .env | grep QUANTUM_KEY | cut -d'=' -f2)
     ```
   - Frontend:
     ```bash
     helm install lawmakers-angular ./helm/lawmakers-angular \
       --set image.repository=your-repo/lawmakers-angular \
       --set image.tag=latest \
       --set env.API_URL=http://lawmakers-api:8000
     ```

5. **Configure Ingress for Networking Hub**:
   - Enable NGINX Ingress controller for secure WebSocket connections (OBS video feeds, private calls).
   - Update `helm/lawmakers-suite/values.yaml`:
     ```yaml
     ingress:
       enabled: true
       hosts:
         - host: lawmakers-suite.your-domain.com
           paths:
             - path: /
               pathType: Prefix
             - path: /hub
               pathType: Prefix
     ```

6. **Enable Metrics and Monitoring**:
   - Install Prometheus and Grafana:
     ```bash
     helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
     helm install prometheus prometheus-community/prometheus
     helm install grafana prometheus-community/grafana
     ```
   - Monitor API latency, database performance, CUDA utilization, and quantum circuit metrics.
   - Access Grafana at `http://grafana.your-domain.com`.

7. **Scale the Deployment**:
   ```bash
   kubectl scale deployment lawmakers-api --replicas=5
   kubectl scale deployment lawmakers-angular --replicas=3
   ```

8. **Verify Deployment**:
   - API: `http://lawmakers-suite.your-domain.com/docs`
   - Frontend: `http://lawmakers-suite.your-domain.com`
   - Networking Hub: `ws://lawmakers-suite.your-domain.com/hub`
   - Jupyter Notebook: `http://lawmakers-suite.your-domain.com:8888`

## Customization Points
- **Cryptographic Modes**:
  - AES-256: Default for data encryption.
  - AES-512: Extended mode for high-security queries (see `crypto_manager.py`).
  - Quantum-Parallel AES-2048: Uses Qiskit for quantum-safe key derivation (see `crypto_manager.py`).
- **Multi-Language Support**: Add MAML templates for additional languages in `maml_processor.maml`.
- **Legal Databases**: Extend `connectors.py` for Bloomberg Law or ROSS Intelligence.
- **GPU Support**: Enable NVIDIA device plugin in Kubernetes for CUDA-based simulations:
  ```bash
  kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/main/nvidia-device-plugin.yml
  ```

## Troubleshooting
- **Pod Failures**: Check logs with `kubectl logs -l app=lawmakers-suite`.
- **WebSocket Issues**: Verify connectivity with `wscat -c ws://lawmakers-suite.your-domain.com/hub`.
- **Cryptography Errors**: Ensure `AES_KEY` is 32 bytes (AES-256) or 64 bytes (AES-512), and `QUANTUM_KEY` is valid (see `crypto_manager.py`).