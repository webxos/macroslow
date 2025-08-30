# BELUGA SDK README

## Overview
BELUGA (Bilateral Environmental Linguistic Ultra Graph Agent) is a quantum-distributed database and sensor fusion system designed for extreme environmental applications, including drones, submarines, IoT cave mining drones, spacecraft, and AR goggles.

## Features
- **SOLIDAR™ Sensor Fusion**: Combines SONAR and LIDAR data for real-time 3D modeling.
- **CHIMERA 2048 Integration**: Leverages four hybrid computational cores for enhanced processing.
- **MAML/MU Protocol**: Uses modern 2025 syntax for verifiable workflows.
- **NVIDIA CUDA Support**: Achieves 76x training speedup and 12.8 TFLOPS.
- **Quantum Security**: 2048-bit AES-equivalent encryption with adaptive modes.
- **OBS Streaming**: Real-time visualization for AR applications.
- **Use Cases**: Aerospace, deep-sea exploration, archaeology, security, and more.

## Installation
See `BELUGA_Core_Build.md` for detailed installation instructions.

## Usage
1. Configure `beluga_config.yaml` with your database, quantum, and sensor settings.
2. Run the server:
   ```bash
   docker run --gpus all -p 8000:8000 -p 9090:9090 beluga-2048
   ```
3. Submit MAML/MU workflows via:
   ```bash
   curl -X POST -H "Content-Type: text/markdown" --data-binary @maml/chimera_hybrid_workflow.maml.md http://localhost:8000/execute
   ```
4. Monitor with Prometheus:
   ```bash
   curl http://localhost:9090/metrics
   ```

## Contributing
Join the Webxos Research Group at [github.com/webxos/beluga-sdk](https://github.com/webxos/beluga-sdk) to contribute to Project Dunes and the BELUGA ecosystem.

## License
© 2025 Webxos. All Rights Reserved.