# 🏜️ MACROSLOW Home Lab Guide 2025: Page 3 – Software Stack Overview

This page presents the **unified software stack** powering all three **MACROSLOW 2048-AES home server builds**—**Minimalist DUNES**, **Chimera GH200**, and **Glastonbury Medical**—through the **PROJECT DUNES** ecosystem. A single, modular framework adapts to legacy low-end systems, overclocked GH200 superchips, and HIPAA-compliant medical environments, using **MAML (Markdown as Medium Language)** for secure, executable workflows and **2048-bit AES-equivalent encryption** across quantum, AI, HPC, and healthcare domains.

## 🧬 Core Operating Environment
- **Base OS**: Ubuntu Server 24.04 LTS (ARM64 for GH200/Orin, x86_64 for legacy)
  - Kernel: 6.8+ with real-time patches (PREEMPT_RT for medical latency)
  - Filesystem: ext4 (Minimalist), ZFS RAID-Z1 (Chimera), encrypted LUKS+ZFS (Glastonbury)
  - Init: systemd with hardened security profiles

## 🚀 Unified SDK Ecosystem

### 1. DUNES 2048 SDK (All Builds – Core Framework)
- **Version**: 3.1.0
- **Purpose**: Orchestrates quantum, AI, HPC, and edge tasks via MAML
- **Modules**:
  - **DUNES Core**: Task scheduler, NVLink-aware resource manager
  - **MAML Runtime**: Compiles `.maml` to Python/ARM assembly
  - **Self-Healing Engine**: Quad-segment regeneration (<5s recovery)
- **Compatibility**:
  - Minimalist: CPU fallback, Qiskit-Aer CPU
  - Chimera: GH200 NVLink-C2C, HBM3e coherence
  - Glastonbury: SED-aware encryption hooks

### 2. Quantum Computing Layer
- **Qiskit 1.3.0**:
  - Backends: Aer (CPU/GPU), Aer-GPU (CUDA 12.4), Braket plugin (optional cloud)
  - Minimalist: 2–8 qubit noisy simulation
  - Chimera: 32–128 qubit hybrid with GH200 tensor cores
  - Glastonbury: Quantum ML for drug interaction modeling
- **Extensions**: Qiskit-Metal (RF sim), Qiskit-Finance (risk analysis)

### 3. AI & HPC Layer
- **PyTorch 2.5.0**:
  - CUDA: 12.4 (GH200), 11.8 (legacy)
  - Precision: FP8 (GH200), BF16 (Orin), FP32 (Minimalist)
  - Chimera: 70B+ LLM training, 10X speedup via HBM coherence
  - Glastonbury: Medical imaging (U-Net, Diffusion), DICOM preprocessing
- **TensorRT 10.0**: Inference optimization (4.5X on GH200)
- **NCCL 2.22**: Multi-node communication (NVLink Network for Chimera)

### 4. MAML Protocol (Executable Markdown)
- **Version**: 2.0
- **Encryption**: 2048-bit AES-equivalent per block
- **Structure**:
'
  ```maml
  ```quantum ... ```
  ```ai ... ```
  ```hpc ... ```
  ```medical ... ```
  ```
- **Compilers**: maml-to-py, maml-to-arm, maml-to-hipaa (Glastonbury)

### 5. Edge & IoT Stack
- **BELUGA Agent 4.0**:
  - Minimalist: Single Pi 4 sensor hub
  - Chimera: Optional Pi 5 telemetry
  - Glastonbury: 3x Pi 5 cluster for wearables (ECG, SpO2)
- **Protocols**: MQTT 5.0 (TLS), WebSocket Secure, CoAP
- **Drivers**: GPIO, I2C, SPI, USB HID (medical devices)

### 6. Medical & Compliance Layer (Glastonbury Exclusive)
- **HIPAA Engine**:
  - Audit logging: Immutable append-only journal
  - Data at Rest: LUKS2 + dm-crypt (AES-XTS 512-bit)
  - PHI Redaction: Auto-masking via DICOM SR
- **DICOM Toolkit**: pydicom 2.4, Orthanc server (local PACS)
- **FHIR Gateway**: FastAPI + OAuth2 for EHR integration

### 7. API & Orchestration
- **FastAPI 0.118**:
  - Async endpoints: /quantum/run, /ai/infer, /medical/fuse
  - Rate limiting: 1000 req/s (Chimera), 100 req/s (Glastonbury)
- **Nginx 1.27**: Reverse proxy, TLS 1.3, HTTP/3
- **Docker 28.0 + Compose 2.30**: Containerized services
  - Chimera: Multi-node Kubernetes-lite (k3s optional)

### 8. Monitoring & Optimization
- **Prometheus 2.54**:
  - Exporters: node, nvidia_gpu, beluga, hipaa_audit
  - Chimera: NVLink bandwidth, HBM utilization
- **Grafana 11.3**:
  - Dashboards: Quantum fidelity, AI throughput, PHI access
- **Optimization Tools**:
  - nsight-systems (GH200 profiling)
  - dcgm (GPU health)
  - zpool iostat (ZFS performance)

## 🔧 Build-Specific Software Profiles

### Minimalist DUNES
- CPU-only Qiskit, PyTorch (no CUDA)
- MAML lightweight parser
- SQLite for workflow logs

### Chimera GH200
- Full GH200 stack: NVLink-C2C, MIG slicing, FP8
- DUNES HPC scheduler with SLURM compatibility
- In-memory datasets (HBM3e)

### Glastonbury Medical
- HIPAA mode enabled in DUNES core
- Encrypted ZFS datasets with key escrow
- BELUGA medical device drivers (FDA Class I compliant)

## 🛡️ Security & Compliance
- **All Builds**: Kernel lockdown, AppArmor, seccomp profiles
- **Encryption**: MAML blocks encrypted at compile time
- **Access Control**: RBAC via FastAPI + JWT
- **Audit**: Immutable logs in /var/log/dunes/audit

## 🔗 Next Steps
Proceed to **Page 4: Assembly & Rackmount Guide** to physically build your chosen server configuration.

*From Legacy to GH200 — Build Your Future with MACROSLOW 2025!* 🏜️
**xAI Artifact Updated**: File `macroslow-homelab-guide-2025.md` updated with Page 3 software stack overview for unified DUNES, Chimera, and Glastonbury SDK ecosystems.
