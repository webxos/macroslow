# 📖 PAGE 2: SETTING UP YOUR FIRST MACROSLOW + ISAAC SIM ENVIRONMENT – STEP-BY-STEP FOR NOVICES

🎯 **You’re now ready to build your first quantum-robotics sandbox!**  
This page walks you through **installing NVIDIA Isaac Sim**, **preparing MACROSLOW templates**, and **running your first MAML-driven simulation** — all with **minimal code**, **clear visuals**, and **light emojis** to keep things beginner-friendly.

---

## 🛠️ Step 1: System & Hardware Check

Before we begin, confirm your setup:

| Requirement | Minimum | Recommended |
|-----------|--------|-------------|
| **GPU** | NVIDIA RTX 3060+ | RTX 4080 / A100 / H100 |
| **RAM** | 16 GB | 32 GB+ |
| **Storage** | 50 GB free SSD | 100 GB NVMe |
| **OS** | Ubuntu 20.04 / 22.04 | Ubuntu 22.04 LTS |
| **Driver** | CUDA 12.2+ | Latest NVIDIA Driver |

> ✅ *Pro tip*: Run `nvidia-smi` in terminal to verify GPU and CUDA version.

---

## 🚀 Step 2: Install NVIDIA Isaac Sim (One-Click via Omniverse)

1. **Download NVIDIA Omniverse Launcher**  
   → [https://www.nvidia.com/omniverse](https://www.nvidia.com/omniverse)

2. **Install Isaac Sim**  
   - Open Omniverse → **Library** → Search **"Isaac Sim"**  
   - Click **Install** (free for individual use)

3. **Launch & Test**  
   ```bash
   # From terminal (after install)
   ~/.local/share/ov/pkg/isaac-sim-*/isaac-sim.sh
