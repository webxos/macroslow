## Page 2: Hardware Selection and Drone Build
The **MACROSLOW SDK** enables Arduino-based drones to evolve from simple PID-controlled quadcopters into **quantum-optimized, self-learning, and agentic swarm systems** using **DUNES**, **CHIMERA**, and **GLASTONBURY** SDKs. This page expands on the introduction by providing **detailed hardware selection**, **drone build guides**, and **use case mappings** across Arduino boards—from legacy **Nano** for basic flight to **Portenta H7** and **GIGA R1 WiFi** for QNN inference, swarm leadership, and real-time MCP/API control. All components are selected for compatibility with **Arduino IDE 2.x**, **MicroPython**, and **MACROSLOW dependencies** (PyTorch, Qiskit, FastAPI), ensuring seamless integration with **2048-AES** security and **quantum simulation offloading** to NVIDIA Jetson Orin (70–275 TOPS).

### Arduino Board Selection for Drone Applications
| **Board** | **Core** | **Clock** | **RAM/Flash** | **Connectivity** | **Best For** | **MACROSLOW Role** |
|-----------|----------|-----------|---------------|------------------|--------------|---------------------|
| **Arduino Nano** | ATmega328P | 16 MHz | 2KB/32KB | UART | Basic flight controller | DUNES sensor logging, offload QNN |
| **Arduino MKR WiFi 1010** | SAMD21 + NINA-W102 | 48 MHz | 32KB/256KB | WiFi/BLE | Mesh follower drones | CHIMERA QKD, real-time API |
| **Arduino Portenta H7** | STM32H747 (M7/M4) | 480/240 MHz | 1MB/2MB | WiFi, ETH, MicroPython | QNN inference, mid-flight training | GLASTONBURY VQE, DUNES MCP |
| **Arduino GIGA R1 WiFi** | Mbed OS (M7/M4) | 1.6 GHz | 1MB/1MB | WiFi/BLE, CAN | Swarm leader, path moderation | CHIMERA API gateway, BELUGA fusion |

> **Recommendation**: Use **Portenta H7** as primary flight controller for QNN and **GIGA R1 WiFi** as swarm leader for MCP coordination.

### Core Drone Components
| **Component** | **Specs** | **Pins** | **Use Case** |
|---------------|-----------|----------|--------------|
| **Brushless Motors** | 2204 2300KV | ESC signal | Propulsion |
| **ESCs** | 30A BLHeli_S | PWM (D3,5,6,9) | Motor speed control |
| **MPU6050** | 6-DOF IMU | I2C (SDA/SCL) | Attitude estimation |
| **VL53L0X** | Time-of-Flight | I2C | Altitude hold |
| **NRF24L01** | 2.4GHz | SPI | Swarm mesh (low latency) |
| **LiPo Battery** | 3S 2200mAh | XT60 | 10–15 min flight |
| **PDB** | Power Distribution | Solder pads | Clean power routing |

### Build Guide: 250mm Quantum Quadcopter
#### Frame Assembly
1. 3D print 250mm X-frame (PLA, 1.5mm walls).
2. Mount motors with M3 screws, align CCW/CW props.
3. Solder ESCs to PDB (battery input → motor outputs).
4. Mount flight controller (Portenta H7) centrally with vibration dampers.

#### Wiring Diagram
```
Battery (3S) → PDB
PDB → ESCs (4x) → Motors
Portenta H7:
  D3, D5, D6, D9 → ESC signal
  SDA/SCL → MPU6050 + VL53L0X
  SPI → NRF24L01 (swarm)
  USB-C → Host (QNN offload)
```

#### Power & Safety
- Use **smoke stopper** during first power-on.
- Balance charge LiPo at 1C.
- Add **low-voltage buzzer** (D2).

### Use Cases and Hardware Mapping
| **Use Case** | **Hardware** | **MACROSLOW SDK** | **Key Feature** |
|--------------|--------------|-------------------|-----------------|
| **1. Autonomous Delivery Drone** | Portenta H7 + VL53L0X | DUNES + GLASTONBURY | QNN for wind-adaptive landing |
| **2. Swarm Surveillance Mesh** | 5x MKR WiFi 1010 + GIGA R1 leader | CHIMERA + BELUGA | MCP swarm formation |
| **3. Mid-Flight Damage Recovery** | Portenta H7 + MPU6050 | DUNES + QNN retrain | Adaptive PID via live data |
| **4. ARACHNID Landing Pod** | GIGA R1 + 8x hydraulic legs | GLASTONBURY VQE | Quantum trajectory optimization |
| **5. DePIN Mapping Swarm** | 10x Nano + NRF24L01 | CHIMERA API | Real-time path moderation |

### MACROSLOW Integration Points
- **DUNES**: Logs IMU/telemetry in `.maml.md` with 256-bit AES.
- **CHIMERA**: Secures ESC commands via QKD-encrypted FastAPI.
- **GLASTONBURY**: Runs PyTorch QNN on Portenta M7 core, offloads VQE to Jetson.
- **MCP**: GIGA R1 broadcasts context (position, battery, mission) to swarm.

### Community Resources
- **MultiWii**: [github.com/multiwii](https://github.com/multiwii) – PID base.
- **YMFC-AL**: Auto-level quad tutorials.
- **Arduino Drone Forum**: [forum.arduino.cc/t/drone](https://forum.arduino.cc) – ESC calibration, IMU fusion.

**Next**: Page 3 covers programming the flight controller with PID loops and ESC control in Arduino IDE 2.x.

**License**: © 2025 WebXOS Research Group. MIT License; fork at [github.com/webxos/macroslow](https://github.com/webxos/macroslow).