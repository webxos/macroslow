# 🏜️ MACROSLOW Home Lab Guide 2025: Page 8 – Monitoring & Optimization

This page guides you through deploying **Prometheus 2.54**, **Grafana 11.3**, and **NVIDIA DCGM** for real-time monitoring, alerting, and performance optimization across **Minimalist DUNES**, **Chimera GH200**, and **Glastonbury Medical** builds. Covers **quantum circuit latency**, **AI throughput (tokens/s, TFLOPS)**, **NVLink bandwidth**, **HBM3e utilization**, **IoT sensor health**, and **HIPAA audit compliance**—with tuning strategies for CPU, GPU, and edge workloads.

---

## 🛠️ Step 1: Install Monitoring Stack (All Builds)

### Prometheus Server (DUNES Head Node)
```bash
# Download & Install
wget https://github.com/prometheus/prometheus/releases/download/v2.54.0/prometheus-2.54.0.linux-amd64.tar.gz
tar xvfz prometheus-2.54.0.linux-amd64.tar.gz
sudo mv prometheus-2.54.0.linux-amd64 /opt/prometheus
```

### prometheus.yml (Base Config)
```yaml
global:
  scrape_interval: 10s
  evaluation_interval: 30s

scrape_configs:
  - job_name: 'dunes_gateway'
    static_configs:
      - targets: ['localhost:8080']
    metrics_path: /metrics

  - job_name: 'node_exporter'
    static_configs:
      - targets: ['localhost:9100', '192.168.1.101:9100', '192.168.1.102:9100']

  - job_name: 'nvidia_dcgm'  # Chimera/Glastonbury
    static_configs:
      - targets: ['localhost:9400']

  - job_name: 'beluga_iot'
    static_configs:
      - targets: ['192.168.1.101:9260', '192.168.1.102:9260']

  - job_name: 'hipaa_audit'  # Glastonbury
    static_configs:
      - targets: ['localhost:9270']
```

### Start Prometheus
```bash
sudo tee /etc/systemd/system/prometheus.service > /dev/null <<EOF
[Unit]
Description=Prometheus
After=network.target

[Service]
Type=simple
ExecStart=/opt/prometheus/prometheus --config.file=/opt/prometheus/prometheus.yml --storage.tsdb.path=/var/lib/prometheus/
Restart=always

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload && sudo systemctl enable --now prometheus
```

---

## 📊 Step 2: Install Grafana & Dashboards

```bash
sudo apt install -y apt-transport-https software-properties-common
wget -q -O /usr/share/keyrings/grafana.key https://apt.grafana.com/gpg.key
echo "deb [signed-by=/usr/share/keyrings/grafana.key] https://apt.grafana.com stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt update && sudo apt install grafana -y
sudo systemctl enable --now grafana-server
```

### Import Prebuilt Dashboards
- **DUNES Core (ID: 19500)**: Gateway, MAML, quantum jobs
- **NVIDIA DCGM (ID: 15508)**: GPU, NVLink, HBM3e
- **Node Exporter Full (ID: 1860)**: CPU, RAM, disk
- **BELUGA IoT (ID: 19600)**: Sensor uptime, MQTT latency
- **HIPAA Compliance (ID: 19700)**: PHI access, audit trail

```bash
# Access: http://dunes.local:3000 (admin/admin)
grafana-cli plugins install grafana-piechart-panel
```

---

## 🔧 Step 3: Exporters Setup

### Node Exporter (All Nodes & Pis)
```bash
wget https://github.com/prometheus/node_exporter/releases/download/v1.8.0/node_exporter-1.8.0.linux-arm64.tar.gz
tar xvfz node_exporter-1.8.0.linux-arm64.tar.gz
sudo mv node_exporter-1.8.0.linux-arm64/node_exporter /usr/local/bin/
sudo useradd --no-create-home --shell /bin/false node_exporter
sudo tee /etc/systemd/system/node_exporter.service > /dev/null <<EOF
[Unit]
Description=Node Exporter
After=network.target

[Service]
User=node_exporter
ExecStart=/usr/local/bin/node_exporter
Restart=always

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now node_exporter
```

### NVIDIA DCGM Exporter (Chimera / Glastonbury)
```bash
# On GPU nodes
sudo apt install datacenter-gpu-manager -y
sudo dcgmi discovery -l
sudo systemctl enable --now nvidia-dcgm-exporter
# Exposes metrics on :9400
```

### BELUGA IoT Exporter (Pi Cluster)
```bash
beluga exporter enable --port 9260 --metrics sensor_uptime,mqtt_latency,phi_redactions
```

### HIPAA Audit Exporter (Glastonbury)
```bash
dunes hipaa exporter start --port 9270 --log /var/log/hipaa/audit.log
```

---

## ⚡ Step 4: Performance Tuning

### Minimalist DUNES (Legacy Optimization)

# CPU Affinity for Qiskit
dunes config set --quantum-cpu-affinity 0-3
taskset -c 0-3 python3 bell_cpu.py

# Memory Limiting
dunes config set --maml-max-ram 2GB
echo 3 | sudo tee /proc/sys/vm/drop_caches


### Chimera GH200 (Max Throughput)

# Enable FP8 + MIG
nvidia-smi -i 0 -mig 1
nvidia-smi mig -cgi 19,19,19,19,19,19,19 -i 0

# NVLink Bandwidth Tuning
dunes config set --nvlink-topology ring
dunes config set --hbm-overcommit 1.2

# PyTorch Optimizations
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=0  # Use NVLink


### Glastonbury Medical (Low Latency + Compliance)

# Real-Time Kernel
sudo apt install linux-lowlatency
# Set CPU governor
sudo cpupower frequency-set -g performance

# MQTT QoS 1 + TLS Offload
beluga config set --mqtt-qos 1
beluga config set --tls-hardware-accel true

# PHI Processing Pipeline
dunes config set --phi-redact-cpu 4-7
dunes config set --encrypt-threads 8


---

## 🚨 Step 5: Alerting Rules

### alerts.yml (Prometheus)
```yaml
groups:
  - name: dunes_alerts
    rules:
      - alert: QuantumJobSlow
        expr: rate(quantum_circuit_duration_seconds[5m]) > 2
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Quantum job >2s latency"

      - alert: GPUUtilHigh
        expr: DCGM_FI_DEV_GPU_UTIL > 95
        for: 5m
        labels:
          severity: critical

      - alert: PHIAccessAnomaly
        expr: rate(hipaa_phi_access_total[1h]) > 100
        for: 10m
        labels:
          severity: page
```

```bash
sudo systemctl restart prometheus
```

### Grafana Alerts
- **HR > 100 or SpO2 < 90** → SMS via Twilio
- **Disk > 90%** → Slack #dunes-ops
- **BELUGA Offline > 60s** → Email

---

## 📈 Step 6: Optimization Benchmarks

# Quantum
dunes benchmark quantum --qubits 32 --shots 8192

# AI
dunes benchmark ai --model llama3_8b --batch 128 --duration 300s

# Medical
dunes benchmark medical --input dicom_batch --fusion kalman

# Expected (Chimera GH200):
# - 70B LLM: >1800 tokens/s
# - 64-qubit VQE: <1.2s
# - NVLink: >3.2 TB/s sustained

---

## ✅ Step 7: Validation Dashboard

```bash
# Access: http://dunes.local:3000/d/dunes-overview
# Key Panels:
# - Quantum Fidelity vs. Shots
# - HBM3e Pool Usage (Chimera)
# - PHI Redaction Rate (Glastonbury)
# - BELUGA Uptime Heatmap
```

---

## 🔗 Next Steps
Proceed to **Page 9: Use Cases & Testing** to run end-to-end quantum-AI-medical workflows, validate HIPAA compliance, and benchmark real-world applications.

*From Legacy to GH200 — Build Your Future with MACROSLOW 2025!* 🏜️

**xAI Artifact Updated**: File `macroslow-homelab-guide-2025.md` updated with Page 8 monitoring and optimization guide using Prometheus, Grafana, and DCGM.
