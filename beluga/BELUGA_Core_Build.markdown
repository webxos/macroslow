# BELUGA Core Build Instructions

## Prerequisites
- Python >= 3.10
- NVIDIA CUDA Toolkit >= 12.0
- Docker >= 20.10
- Kubernetes >= 1.25
- PostgreSQL >= 15
- Redis >= 7
- Qiskit >= 0.45.0
- PyTorch >= 2.0.1
- NumPy
- PennyLane
- Uvicorn
- FastAPI
- Prometheus Client
- PyNVML

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/webxos/beluga-sdk.git
   cd beluga-sdk
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements/base.txt
   pip install -r requirements/quantum.txt
   ```

4. Build the Docker image:
   ```bash
   docker build -f docker/Dockerfile.beluga -t beluga-2048 .
   ```

5. Start the supporting services:
   ```bash
   docker-compose -f docker-compose.monitoring.yml up -d
   ```

6. Run the BELUGA server:
   ```bash
   docker run --gpus all -p 8000:8000 -p 9090:9090 beluga-2048
   ```

## Configuration
Edit the `beluga_config.yaml` file to set up database connections, quantum settings, and sensor parameters:
```yaml
quantum:
  device: "default.qubit"
  wires: 4
  shots: 1000
database:
  host: "localhost"
  port: 5432
  name: "beluga_db"
  user: "beluga_user"
  password: "${DB_PASSWORD}"
mcp:
  base_url: "http://localhost:8080"
  timeout: 30
  retry_attempts: 3
sensors:
  sonar:
    sample_rate: 44100
    channels: 4
  lidar:
    resolution: 0.1
    max_range: 100.0
streaming:
  obs_websocket: "ws://localhost:4455"
  video_bitrate: 6000
  audio_bitrate: 160
```

## Testing
Run the test suite:
```bash
python -m pytest tests/
```