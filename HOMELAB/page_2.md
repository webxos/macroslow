# 🏜️ MACROSLOW Home Lab Guide 2025: Page 2 – Hardware Requirements (All 3 Builds)

This page provides a detailed, in-depth breakdown of hardware components for the three **MACROSLOW 2048-AES home server builds**: **Minimalist DUNES 2048 SDK**, **Chimera 2048-AES SDK with GH200 Overclock**, and **Glastonbury Medical 2048 SDK**. Each configuration is optimized for specific workloads—entry-level quantum simulation, high-performance AI/HPC, and secure healthcare data processing—while maintaining compatibility with the unified **PROJECT DUNES** software stack. All builds support rackmount or desktop deployment, NVLink clustering, and real-time local networking in home or professional settings.

## 1. Minimalist DUNES 2048 SDK (~$200–$500)
Designed for legacy hardware integration, low-power quantum qubit simulation (2–8 qubits), lightweight AI inference, and IoT prototyping on older or entry-level systems.

### Core Compute
- **NVIDIA GPU**: GTX 1060 6GB, GTX 1660 Super, or Jetson Nano 4GB Developer Kit
  - CUDA Compute Capability: 6.1+ (Pascal or newer)
  - Memory: Minimum 4GB GDDR5
  - TDP: 120W max
  - Ports: HDMI 2.0, USB 3.0 Type-A
- **CPU (Host)**: Intel Core i5-6400 or AMD Ryzen 3 1200 (4 cores/8 threads)
  - Socket: LGA 1151 or AM4
  - RAM Support: DDR4 2400 MHz
  - PCIe: 3.0 x16 slot for GPU

### System Board
- **Motherboard**: ASUS Prime B450M-A or Gigabyte B360M DS3H
  - Form Factor: Micro-ATX
  - Expansion: 1x PCIe 3.0 x16, 2x M.2 NVMe, 4x SATA 6Gb/s
  - Networking: Realtek RTL8111H Gigabit LAN
  - Rear I/O: 4x USB 3.1 Gen 1, 2x USB 2.0, PS/2, D-Sub, HDMI

### Memory & Storage
- **RAM**: 16GB (2x8GB) DDR4 2400 MHz CL16
  - Latency: CAS 16-16-16-36
  - Voltage: 1.2V
- **Primary Storage**: 256GB NVMe SSD (Samsung 970 EVO or WD Blue SN550)
  - Interface: PCIe 3.0 x4
  - Read/Write: 3500/3000 MB/s
- **Secondary Storage**: 1TB 2.5" SATA SSD or 2TB 3.5" HDD (5400 RPM)
  - Use: Dataset caching, MAML workflow archives

### Edge & IoT
- **Raspberry Pi**: Raspberry Pi 4 Model B 4GB
  - SoC: Broadcom BCM2711, Quad-core Cortex-A72 @ 1.8 GHz
  - RAM: 4GB LPDDR4-3200
  - Ports: 2x micro HDMI, 2x USB 3.0, 2x USB 2.0, Gigabit Ethernet, 40-pin GPIO
  - Storage: 64GB microSD Class 10

### Power, Cooling & Enclosure
- **PSU**: 450W 80+ Bronze (Corsair CX450M)
  - Connectors: 1x 24-pin ATX, 1x 8-pin EPS, 2x 6+2-pin PCIe
  - Efficiency: 85% at 50% load
- **Cooling**: Stock CPU cooler + 2x 120mm case fans (intake/exhaust)
  - Airflow: 50 CFM each
- **Case**: Fractal Design Node 304 (Mini-ITX) or open-frame 3D-printed chassis
  - Dimensions: 197 x 210 x 374 mm
  - Drive Bays: 6x 3.5"/2.5"

### Networking
- **Switch**: TP-Link TL-SG108 8-port Gigabit unmanaged
  - Backplane: 16 Gbps
  - Ports: 8x RJ45 10/100/1000 Mbps
- **Cables**: 5x Cat6 flat Ethernet (1m–3m)

---

## 2. Chimera 2048-AES SDK + GH200 Overclock (~$5,000–$15,000+)
Engineered for maximum AI/HPC throughput, overclocked **NVIDIA GH200 Grace Hopper Superchip** performance, Colossus 2-style NVLink clustering, and large-scale quantum-classical hybrid simulations.

### Core Compute (Per Node)
- **NVIDIA GH200 Grace Hopper Superchip**: 1–4 units
  - CPU: 72-core Arm Neoverse V2 (Grace)
  - GPU: H100 141GB HBM3e (or 94GB variant)
  - Unified Memory: 96GB HBM3e + 480GB LPDDR5X (coherent via NVLink-C2C)
  - Bandwidth: 900 GB/s (HBM), 8 TB/s (NVLink-C2C)
  - TDP: 700W per module
  - Interconnect: NVLink 5.0 (900 GB/s bidirectional per link)

### System Board
- **Motherboard**: Supermicro H13SSL-NT or NVIDIA DGX GH200 reference board
  - Form Factor: E-ATX
  - Expansion: 4x PCIe 5.0 x16 (NVLink), 8x M.2 NVMe, 16x DDR5 DIMM slots
  - BMC: ASPEED AST2600 with IPMI 2.0

### Memory & Storage
- **RAM**: 512GB–2TB DDR5-5600 ECC Registered (16x 32GB–128GB modules)
  - Bandwidth: 716.8 GB/s aggregate
  - Latency: CL46
- **Primary Storage**: 4x 3.84TB NVMe SSD (Micron 7450 PRO) in RAID 0
  - Read/Write: 6800/5600 MB/s per drive
  - Endurance: 1.3 DWPD
- **Cache Tier**: 2x 1.92TB Optane SSD (optional for write-intensive AI training)

### Power, Cooling & Enclosure
- **PSU**: 2000W 80+ Titanium redundant (2x 1000W modules)
  - Connectors: 4x 8-pin CPU, 8x 8-pin PCIe
  - Efficiency: 96% at 50% load
- **Cooling**: Custom liquid cooling loop with 360mm + 480mm radiators, 8x high-static-pressure fans
  - Coolant: EK-CryoFuel or equivalent
  - Pump: Dual D5 in series
- **Case**: 4U rackmount chassis (Rosewill RSV-L4412U or Supermicro CSE-847)
  - Dimensions: 178 x 437 x 699 mm
  - Drive Bays: 36x 3.5" hot-swap

### Networking
- **Switch**: NVIDIA Mellanox SN3700C 32-port 200GbE
  - Backplane: 6.4 Tbps
  - Ports: 32x QSFP56 (InfiniBand or Ethernet)
- **Interconnect**: NVLink Network for multi-node (900 GB/s per GPU pair)
- **Cables**: 4x QSFP56 DAC (1m), 8x Cat8 Ethernet (10GbE fallback)

---

## 3. Glastonbury Medical 2048 SDK (~$1,000–$3,000)
Tailored for HIPAA-compliant, local healthcare data processing in small clinics, physician offices, or home health labs. Focus on encrypted storage, medical imaging AI, and wearable data fusion.

### Core Compute
- **NVIDIA GPU**: Jetson AGX Orin 64GB or RTX 4070 12GB
  - CUDA Cores: 2048 (Orin) or 5888 (RTX 4070)
  - Tensor Cores: 64 (Orin) or 184 (RTX 4070)
  - Memory: 64GB LPDDR5 (Orin) or 12GB GDDR6X
  - TDP: 60W (Orin) or 200W
- **CPU (Host)**: Intel Core i7-13700 or AMD Ryzen 7 7700
  - Cores/Threads: 16/24 or 8/16
  - Cache: 30MB or 40MB

### System Board
- **Motherboard**: ASUS ProArt B650-Creator or MSI MAG B760M Mortar
  - Form Factor: Micro-ATX
  - Expansion: 1x PCIe 5.0 x16, 2x M.2 NVMe (encrypted), 4x SATA
  - Security: TPM 2.0 header, BIOS-level encryption enablement

### Memory & Storage
- **RAM**: 32GB (2x16GB) DDR5 6000 MHz CL36
  - ECC: Optional (for data integrity)
- **Primary Storage**: 2x 2TB NVMe SSD (Samsung 990 PRO) in RAID 1 (encrypted)
  - Encryption: Hardware AES-256 (SED)
  - Read/Write: 7450/6900 MB/s
- **Archive Storage**: 8TB HDD (WD Red Plus) in encrypted enclosure
  - RPM: 7200
  - Interface: SATA 6Gb/s

### Edge & IoT
- **Raspberry Pi Cluster**: 3x Raspberry Pi 5 8GB
  - SoC: Broadcom BCM2712, Quad-core Cortex-A76 @ 2.4 GHz
  - RAM: 8GB LPDDR4X-4267
  - Ports: 2x USB 3.0, Gigabit Ethernet, 40-pin GPIO
  - Storage: 128GB microSD per node

### Power, Cooling & Enclosure
- **PSU**: 650W 80+ Gold (Seasonic Focus GX-650)
  - Connectors: 1x 24-pin, 1x 8-pin EPS, 2x 6+2-pin PCIe
- **Cooling**: Noctua NH-U12S CPU cooler + 3x 92mm case fans
  - Noise: <25 dBA
- **Case**: Silverstone CS351 (NAS-style) or 2U rackmount
  - Drive Bays: 5x 3.5" hot-swap
  - Security: Lockable front panel

### Networking
- **Switch**: QNAP QSW-1105-5T 5-port 2.5GbE unmanaged
  - Backplane: 25 Gbps
  - Ports: 5x 2.5GBASE-T
- **Cables**: 6x Cat6a shielded (1m–5m)
- **VLAN**: Dedicated medical data VLAN (ID 100)

---

## Shared Hardware Considerations
- **Rackmount Compatibility**: All builds support 19" EIA-310 racks; use rail kits for 2U/4U chassis.
- **Power Backup**: UPS with 1500VA+ (APC Smart-UPS) for graceful shutdown.
- **Thermal Environment**: 18–27°C operating temp; avoid direct sunlight.
- **Cable Management**: Velcro ties, labeled ports, color-coded Ethernet.

## Next Steps
Proceed to **Page 3: Software Stack Overview** to explore the unified DUNES, Chimera, and Glastonbury SDK ecosystems.

*From Legacy to GH200 — Build Your Future with MACROSLOW 2025!* 🏜️
