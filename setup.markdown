# Setting Up the GLASTONBURY 2048 SDK

## Prerequisites

- **Software**:
  - Python 3.8+, Qiskit, PyTorch, SQLAlchemy, FastAPI
  - Angular CLI (`npm install -g @angular/cli`)
  - Xcode with Core Bluetooth and Find My SDK
- **Hardware**:
  - Apple Watch (Series 6+, watchOS 8+)
  - Apple AirTags
  - Nordic nRF52840 Bluetooth Mesh nodes
  - CommScope SYSTIMAX tether cables
- **Accounts**:
  - Apple Developer Program for Core Bluetooth and Find My access
  - IBM Quantum or AWS Braket for quantum processing (optional)

## Installation

1. **Python Dependencies**:
   ```bash
   pip install qiskit torch sqlalchemy fastapi uvicorn python-jose[cryptography] cryptography
   ```

2. **AngularJS Setup**:
   ```bash
   ng new glastonbury-dashboard
   cd glastonbury-dashboard
   npm install angular
   ```

3. **Bluetooth Mesh Nodes**:
   Flash nRF52840 nodes with Zephyr RTOS:
   ```bash
   west init mesh-project
   cd mesh-project
   west update
   west build -b nrf52840dk_nrf52840 zephyr/samples/bluetooth/mesh
   west flash
   ```

4. **Apple Watch & AirTag Setup**:
   Configure Core Bluetooth and Find My in Xcode:
   ```swift
   import CoreBluetooth
   import FindMyDevice
   class DeviceManager: NSObject, CBCentralManagerDelegate {
       var centralManager: CBCentralManager!
       override init() {
           super.init()
           centralManager = CBCentralManager(delegate: self, queue: nil)
       }
       func centralManagerDidUpdateState(_ central: CBCentralManager) {
           if central.state == .poweredOn { print("Bluetooth Mesh ready!") }
       }
   }
   ```

## Next Steps

- Configure the FastAPI server (`server_setup.md`).
- Set up hardware (`hardware_guide.md`).