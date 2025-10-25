# 🐉 CHIMERA 2048-AES Homelab: Page 5 – Software Installation and Configuration

This page guides you through installing and configuring the software stack for your **CHIMERA 2048-AES Homelab** across Budget, Mid-Tier, and High-End builds. The process includes setting up **Ubuntu**, **NVIDIA CUDA**, and **MACROSLOW CHIMERA 2048-AES SDK** dependencies to support quantum, AI, and IoT workloads.

## 🛠️ Installation Steps

### 1. Install Ubuntu 24.04 LTS
- **All Builds**:
  1. Download Ubuntu 24.04 LTS ISO from [ubuntu.com](https://ubuntu.com).
  2. For Jetson/GPU (Budget/Mid-Tier/High-End):
     - Flash ISO to USB drive using Balena Etcher.
     - Boot from USB and install Ubuntu on SSD/NVMe.
  3. For Raspberry Pi:
     - Flash Ubuntu 24.04 (server) to microSD/NVMe using Raspberry Pi Imager.
     - Insert microSD/NVMe into Pi, boot, and complete setup via SSH.
  4. Update system: `sudo apt update && sudo apt upgrade -y`.

### 2. Install NVIDIA CUDA Toolkit
- **Budget (Jetson Nano)**:
  1. Install JetPack 5.1.3 via NVIDIA SDK Manager.
  2. Enable CUDA 11.8: `sudo apt install nvidia-jetpack -y`.
- **Mid-Tier (Jetson AGX Orin)**:
  1. Install JetPack 6.0 with CUDA 12.2 via SDK Manager.
  2. Verify: `nvcc --version`.
- **High-End (RTX A6000/A100)**:
  1. Download CUDA 12.2 from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com).
  2. Install: `sudo bash cuda_12.2.0_535.86.10_linux.run`.
  3. Install cuDNN 9.0: Follow NVIDIA instructions.
  4. Verify: `nvidia-smi` and `nvcc --version`.

### 3. Install CHIMERA 2048-AES SDK Dependencies
- **All Builds**:
  1. Install Python 3.11: `sudo apt install python3.11 python3-pip -y`.
  2. Install Docker: `sudo apt install docker.io docker-compose -y`.
  3. Install dependencies:
     ```bash
     sudo apt install build-essential libssl-dev libffi-dev python3-dev -y
     pip3 install numpy scipy requests
