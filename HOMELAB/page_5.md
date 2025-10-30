# 🏜️ MACROSLOW Home Lab Guide 2025: Page 5 – Installation & Networking

This page provides a comprehensive, step-by-step guide to installing **Ubuntu Server 24.04 LTS**, **NVIDIA drivers**, **CUDA**, **DUNES SDK dependencies**, and configuring **secure, high-performance networking** across all three **MACROSLOW 2048-AES builds**: **Minimalist DUNES**, **Chimera GH200**, and **Glastonbury Medical**. Networking includes VLAN segmentation, 10GbE/2.5GbE fabrics, NVLink-C2C clustering (Chimera), and HIPAA-isolated medical traffic.

---

## 🛠️ OS Installation (All Builds)

### Step 1: Prepare Boot Media
1. Download Ubuntu Server 24.04 LTS:
   - **x86_64**: Standard ISO
   - **ARM64**: For GH200, Jetson AGX Orin
2. Flash to USB (≥16GB) using **Rufus** (Windows) or `dd` (Linux):
   ```bash
   sudo dd if=ubuntu-24.04-server-arm64.iso of=/dev/sdX bs=4M status=progress && sync
   ```

### Step 2: Initial Boot & Partitioning
1. Insert USB, boot via BIOS/UEFI (F2/Del)
2. **Minimalist / Glastonbury**:
   - Full disk install on primary NVMe
   - Filesystem: ext4
   - Enable LUKS encryption (Glastonbury only)
3. **Chimera GH200**:
   - Manual partitioning:
     - `/boot/efi`: 1GB FAT32
     - `/boot`: 2GB ext4
     - `/`: ZFS root (RAID-Z1 across 4x NVMe)
   - Enable ZFS encryption with passphrase
4. Set hostname:
   - Minimalist: `dunes-node1`
   - Chimera: `chimera-node1`, `chimera-node2`, etc.
   - Glastonbury: `glaston-node1`

### Step 3: Post-Install Updates
```bash
sudo apt update && sudo apt full-upgrade -y
sudo apt install linux-headers-$(uname -r) build-essential dkms -y
```

---

## 🚀 NVIDIA & CUDA Installation

### Minimalist DUNES (GTX 1060 / Jetson Nano)
```bash
# Legacy GPU
sudo add-apt-repository ppa:graphics-drivers/ppa -y
sudo apt install nvidia-driver-470 -y

# Jetson Nano (via JetPack)
sudo apt install nvidia-jetpack -y  # CUDA 11.8
```

### Chimera GH200 (Grace Hopper)
```bash
# ARM64 CUDA 12.4 (GH200)
wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.15_linux_arm64.run
sudo sh cuda_12.4.0_550.54.15_linux_arm64.run --silent --toolkit

# Verify NVLink-C2C
nvidia-smi nvlink -i 0 -s
```

### Glastonbury Medical (RTX 4070 / AGX Orin)
```bash
# RTX 4070
sudo apt install nvidia-driver-560 -y
# CUDA 12.2
ubuntu-drivers install

# Jetson AGX Orin
sudo apt install nvidia-jetpack -y  # CUDA 12.2
```

### All Builds: Common CUDA Setup
```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
sudo ldconfig
```

---

## 🧪 DUNES SDK & Core Dependencies

```bash
# Python & Tools
sudo apt install python3.11 python3-pip python3-venv git curl wget -y

# Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Clone DUNES Repo
git clone https://github.com/webxos/dunes.git ~/dunes
cd ~/dunes

# Install SDK (Unified for All Builds)
pip3 install -r requirements.txt
python3 setup.py install

# Initialize Config
dunes init --profile minimalist  # or chimera / glastonbury
```

### Build-Specific Profiles
- **Minimalist**: `--profile minimalist` → CPU-only Qiskit, no NVLink
- **Chimera**: `--profile chimera` → NVLink scheduler, HBM3e awareness
- **Glastonbury**: `--profile glastonbury` → HIPAA logging, LUKS hooks

---

## 🌐 Networking Configuration

### Step 1: Static IP & Netplan (All Builds)
Edit `/etc/netplan/01-netcfg.yaml`:
```yaml
network:
  version: 2
  ethernets:
    enp3s0:  # Main NIC
      dhcp4: no
      addresses: [192.168.1.10/24]
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8, 1.1.1.1]
  vlans:
    vlan.10:
      id: 10
      link: enp3s0
      addresses: [10.0.10.10/24]
    vlan.20:
      id: 20
      link: enp3s0
      addresses: [10.0.20.10/24]
    vlan.100:  # Medical Isolated
      id: 100
      link: enp3s0
      addresses: [172.16.100.10/24]
```
```bash
sudo netplan apply
```

### Step 2: Switch Configuration

#### Minimalist (TP-Link 8-port)
- Plug & Play
- All ports in default VLAN

#### Chimera (Mellanox SN3700C 200GbE)
```bash
# Enable NVLink over Ethernet (RoCE)
enable
configure terminal
interface ethernet 1/1
  mtu 9000
  speed 200G
  fec rs
  dcbx ieee
exit
```
- **VLAN 10**: Management (IPMI)
- **VLAN 20**: AI Data (NVLink + RoCE)
- **Jumbo Frames**: MTU 9000

#### Glastonbury (QNAP 2.5GbE)
- Port 1: Server
- Port 2–4: Pi Cluster
- Port 5: **Isolated Medical VLAN 100** (no internet)

---

## 🔒 Security & Compliance Hardening

### All Builds
```bash
# Firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow from 192.168.1.0/24 to any port 22  # SSH
sudo ufw allow 80,443,8080
sudo ufw enable

# SSH Hardening
sudo sed -i 's/#PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i 's/#PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart ssh
```

### Glastonbury Medical (HIPAA)
```bash
# LUKS Encryption
sudo cryptsetup luksFormat /dev/nvme0n1p3
sudo cryptsetup luksOpen /dev/nvme0n1p3 cryptroot

# Auditd
sudo apt install auditd audispd-plugins -y
sudo systemctl enable auditd

# Immutable Logs
sudo chattr +a /var/log/hipaa_audit.log
```

---

## 🖧 Multi-Node Clustering (Chimera)

```bash
# Node 1 (Head)
dunes cluster init --nodes 4 --fabric nvlink

# Nodes 2–4
dunes cluster join --token <token> --head 10.0.10.10

# Verify
dunes cluster status
# Output: 4 nodes, 16 H100 GPUs, 3.6 TB/s NVLink
```

---

## ✅ Final Verification

```bash
# GPU
nvidia-smi

# CUDA
nvcc --version

# DUNES
dunes --version

# Network
ip a | grep vlan
ping -c 3 10.0.20.11  # Cross-node

# Glastonbury: Encryption
lsblk -f | grep crypto
```

## 🔗 Next Steps
Proceed to **Page 6: SDK Setup (DUNES/Chimera/Glastonbury)** to deploy the unified gateway, MAML workflows, and build-specific optimizations.

*From Legacy to GH200 — Build Your Future with MACROSLOW 2025!* 🏜️
