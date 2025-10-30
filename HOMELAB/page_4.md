# 🏜️ MACROSLOW Home Lab Guide 2025: Page 4 – Assembly & Rackmount Guide

This page delivers a comprehensive, step-by-step physical assembly guide for all three **MACROSLOW 2048-AES home server builds**—**Minimalist DUNES**, **Chimaera GH200**, and **Glastonbury Medical**—covering desktop, open-frame, and professional **19-inch rackmount** configurations. Each build includes detailed procedures for component installation, thermal management, power sequencing, cable routing, and multi-node clustering (NVLink/10GbE), ensuring operational stability in home labs, small offices, or medical environments.

---

## 🛠️ General Preparation (All Builds)

### Workspace & Safety
- **Environment**: Clean, ESD-safe, well-lit area (18–25°C, 40–60% humidity)
- **Tools**:
  - Phillips #1/#2 screwdrivers
  - Anti-static wrist strap (grounded)
  - Thermal paste (Arctic MX-6 or Noctua NT-H2)
  - Cable ties (Velcro preferred), labels, torque driver (1.5 Nm)
  - Digital multimeter (for PSU testing)
- **Safety**:
  - Disconnect all power before handling components
  - Wear ESD strap connected to chassis ground
  - Avoid loose clothing near fans

---

## 1. Minimalist DUNES 2048 SDK – Desktop/Open-Frame Assembly

### Step 1: Motherboard & CPU Installation
1. Place ASUS Prime B450M-A in ESD bag as work surface
2. Install Intel i5-6400 or Ryzen 3 1200 into socket
   - Align triangle marker, lower lever gently
3. Apply pea-sized thermal paste to CPU IHS
4. Mount stock cooler:
   - Secure with push-pins (diagonal order)
   - Connect 4-pin PWM to CPU_FAN header

### Step 2: RAM & Storage
1. Insert 2x8GB DDR4 into slots A2/B2 (dual-channel)
   - Click firmly until latches engage
2. Install 256GB NVMe SSD in primary M.2 slot
   - Secure with captive screw
3. Connect 1TB SATA SSD/HDD to SATA0 port + power

### Step 3: GPU & Case Integration
1. Remove PCIe slot covers from case
2. Insert GTX 1060/1660 Super into PCIe x16
   - Secure with thumb screw
   - Connect 6+2-pin PCIe power
3. Mount Jetson Nano (optional) via USB 3.0 + HDMI
4. Install in Fractal Node 304 or 3D-printed open frame
   - Secure motherboard with 6–9 standoffs
   - Route front panel connectors (PWR_SW, RESET_SW, HDD_LED)

### Step 4: Raspberry Pi 4 Integration
1. Flash 64GB microSD with Ubuntu
2. Mount Pi 4 on acrylic standoffs inside case
3. Connect via Gigabit Ethernet to motherboard LAN
4. Power via USB-C from motherboard header (5V/3A)

### Step 5: Power & Final Checks
1. Connect Corsair CX450M:
   - 24-pin ATX → Motherboard
   - 8-pin EPS → CPU
   - 6+2-pin → GPU
   - SATA power → SSD/HDD
2. Install 2x 120mm case fans (intake front, exhaust rear)
3. Cable management: Bundle with Velcro, route behind tray
4. Initial POST:
   - Connect HDMI + USB keyboard
   - Power on → Verify BIOS boot, GPU detection

---

## 2. Chimera 2048-AES SDK + GH200 – 4U Rackmount Cluster Assembly

### Step 1: Chassis & Rail Preparation
1. Install Rosewill RSV-L4412U in 19" rack
   - Use universal rail kit, secure at 4 points per side
   - Ensure 1U clearance above/below for airflow
2. Ground chassis to rack frame

### Step 2: GH200 Node Installation (Per Node)
1. Insert Supermicro H13SSL-NT into 4U tray
2. Install 72-core Grace CPU (pre-soldered on GH200 module)
3. Populate 16x DDR5 ECC RDIMM (32GB–128GB each)
   - Slots 1–16 in order, click until locked
4. Mount 4x 3.84TB NVMe in front hot-swap bays
   - Secure with tool-less caddies
5. Connect NVLink-C2C bridges between GH200 modules (if multi-GPU/node)

### Step 3: Liquid Cooling Loop (Per Node)
1. Install 360mm radiator (push/pull, 6x 120mm fans)
2. Mount dual D5 pumps in series on reservoir bracket
3. Route soft tubing:
   - CPU/GPU block → Radiator → Reservoir → Pump
4. Fill with EK-CryoFuel, bleed air for 30 mins
5. Connect pump tach to CPU_OPT, fans to SYS_FAN headers

### Step 4: Power & Multi-Node Interconnect
1. Install 2x 1000W Titanium PSU (redundant)
   - Connect via PDB (Power Distribution Board)
   - 4x 8-pin CPU, 8x 8-pin PCIe per PSU
2. **NVLink Network (Multi-Node)**:
   - Connect GH200 GPU0 → GPU1 via NVLink 5.0 bridge (900 GB/s)
   - Node-to-node: QSFP56 DAC cables (900 GB/s bidirectional)
3. **Control Plane**: 10GbE management via BMC (IPMI)

### Step 5: Final Rack Integration
1. Slide nodes into 4U chassis (rail lock)
2. Connect Mellanox SN3700C switch:
   - 200GbE QSFP56 uplinks to each node
   - VLAN 10 (control), VLAN 20 (data)
3. Power sequencing:
   - UPS → PDU → PSUs
   - Staggered startup (10s delay per node)

---

## 3. Glastonbury Medical 2048 SDK – 2U Rackmount / Desktop Hybrid

### Step 1: Motherboard & CPU
1. Install ASUS ProArt B650 or MSI B760M in Silverstone CS351
2. Mount i7-13700 / Ryzen 7 7700
   - Apply thermal paste in 5-dot pattern
   - Secure Noctua NH-U12S with backplate

### Step 2: Encrypted Storage Array
1. Install 2x 2TB Samsung 990 PRO in M.2 slots
   - Enable TCG Opal 2.0 in BIOS
2. Mount 8TB WD Red Plus in 3.5" bay
   - Connect via SATA + Molex power
3. Install 3x Pi 5 in internal cluster tray
   - Acrylic mount with 40mm fans
   - Ethernet daisy-chain via onboard switch

### Step 3: GPU & Medical I/O
1. Insert RTX 4070 or Jetson AGX Orin in PCIe x16
   - Secure with locking bracket
   - Connect 8-pin PCIe power
2. Route medical device ports:
   - USB 3.2 Gen 2 (ECG, pulse oximeter)
   - RS-232 adapter (legacy infusion pumps)

### Step 4: Networking & Security
1. Connect QNAP 2.5GbE switch:
   - Port 1: Server
   - Port 2–4: Pi cluster
   - Port 5: Isolated medical VLAN (100)
2. Physical security:
   - Lockable front bezel
   - Kensington lock on chassis

### Step 5: Power & Validation
1. Connect Seasonic 650W PSU
   - 24-pin, 8-pin EPS, 6+2-pin PCIe
2. UPS integration (APC 1500VA)
3. Boot to BIOS:
   - Enable TPM 2.0, Secure Boot
   - Set boot order: NVMe → USB

---

## 🔗 Multi-Build Integration & Clustering

### NVLink-C2C (Chimera Only)
- Max 4 GH200 nodes in ring topology
- Bandwidth: 3.6 TB/s aggregate
- Software: NCCL over NVLink Network

### 10GbE/2.5GbE Fabric (All Builds)
- VLAN segmentation:
  - 10: Management
  - 20: Data/AI
  - 100: Medical (Glastonbury)
- Jumbo frames (MTU 9000) for AI datasets

### Cable Management Standards
- Color coding:
  - Blue: Power
  - Black: Data (NVLink/Ethernet)
  - Red: Medical/IoT
- Labeling: Node ID, port function, cable length

---

## ⚡ Final Power-On Sequence
1. UPS → PDU → PSUs
2. Management switch → Control plane
3. Staggered node boot (10s intervals)
4. Verify:
   - BMC/IPMI accessible
   - GPU detected (`nvidia-smi`)
   - ZFS pool online
   - BELUGA agents registered

## Next Steps
Proceed to **Page 5: Installation & Networking** to configure Ubuntu, CUDA, and secure network fabrics.

*From Legacy to GH200 — Build Your Future with MACROSLOW 2025!* 🏜️
