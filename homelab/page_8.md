# 🐉 CHIMERA 2048-AES Homelab: Page 7 – Configuring Raspberry Pi for IoT and Edge Tasks

This page details the setup and configuration of **Raspberry Pi** units in your **CHIMERA 2048-AES Homelab** for IoT and edge computing tasks using the **BELUGA Agent**. These steps apply to Budget (1x Pi 4), Mid-Tier (3x Pi 5), and High-End (5x Pi 5) builds, enabling sensor fusion and real-time data processing.

## 🛠️ Configuration Steps

### 1. Prepare Raspberry Pi OS
- **All Builds**:
  1. Ensure Ubuntu 24.04 (server) is installed on each Pi (see Page 5).
  2. Update system: `sudo apt update && sudo apt upgrade -y`.
  3. Set hostname for each Pi (e.g., `pi1`, `pi2`): `sudo hostnamectl set-hostname pi1`.
  4. Configure static IPs in `/etc/netplan/01-netcfg.yaml`:
     ```yaml
     network:
       ethernets:
         eth0:
           addresses: [192.168.1.10/24]
           gateway4: 192.168.1.1
           nameservers:
             addresses: [8.8.8.8, 8.8.4.4]
     ```
  5. Apply: `sudo netplan apply`.

### 2. Install BELUGA Agent
- **Steps**:
  1. Install dependencies: `sudo apt install mosquitto mosquitto-clients python3-pip -y`.
  2. Install BELUGA Agent: `pip3 install beluga-agent==3.0`.
  3. Initialize BELUGA: `beluga init --config beluga.yaml`.
  4. Edit `beluga.yaml`:
     ```yaml
     agent:
       id: "pi1"
       mqtt:
         broker: "localhost:1883"
         topic: "iot/sensors"
       tasks:
         sensor_fusion: true
         output: "file:///results/iot_data.json"
     ```
  5. Start agent: `beluga start --config beluga.yaml`.

### 3. Configure IoT Sensor Integration
- **Steps**:
  1. Connect sensors (e.g., temperature, motion) to Pi GPIO or USB ports.
  2. Install sensor libraries (e.g., for DHT22): `pip3 install adafruit-circuitpython-dht`.
  3. Create a sensor script (`sensor.py`):
     ```python
     import adafruit_dht
     import paho.mqtt.client as mqtt
     from board import D4
     dht = adafruit_dht.DHT22(D4)
     client = mqtt.Client()
     client.connect("localhost", 1883)
     while True:
         temp = dht.temperature
         client.publish("iot/sensors", f"temp: {temp}")
     ```
  4. Run: `python3 sensor.py`.

### 4. Integrate with CHIMERA Gateway
- **Steps**:
  1. Update `chimera.yaml` on main server to include Pi agents:
     ```yaml
     iot:
       agents:
         - id: "pi1"
           endpoint: "mqtt://192.168.1.10:1883"
         - id: "pi2"
           endpoint: "mqtt://192.168.1.11:1883"
     ```
  2. Restart gateway: `chimera gateway restart --config chimera.yaml`.
  3. Test integration: `chimera iot status` (should list active Pi agents).

### 5. Test Sensor Fusion
- **Steps**:
  1. Create a MAML workflow for sensor fusion (`iot.maml`):
     ```markdown
     # IoT Sensor Fusion
     ```iot
     task: sensor_fusion
     agents: ["pi1", "pi2"]
     input: mqtt://localhost:1883/iot/sensors
     output: file:///results/fused_data.json
     ```
  2. Validate and run: `chimera maml validate iot.maml && chimera maml run iot.maml`.
  3. Verify output: Check `/results/fused_data.json` for aggregated sensor data.

## 💡 Tips for Success
- **Networking**: Ensure Pis are on the same VLAN as the main server.
- **Sensor Calibration**: Test sensors individually before fusion.
- **Logs**: Monitor BELUGA logs at `~/.beluga/logs` for errors.
- **Scalability**: Add more Pis for Mid/High-End builds by repeating steps.

## ⚠️ Common Issues
- **MQTT Failure**: Verify Mosquitto is running (`sudo systemctl status mosquitto`).
- **Sensor Errors**: Check GPIO pin assignments and library compatibility.
- **Network Latency**: Ensure stable Ethernet/Wi-Fi for real-time data.

## 🔗 Next Steps
Proceed to **Page 8: Monitoring and Optimization** to set up Prometheus and CUDA tools for performance tracking.

*Unleash the Quantum Beast with CHIMERA 2048 and WebXOS 2025!* 🐉

**xAI Artifact Updated**: File `readme.md` updated with Page 7 content for CHIMERA 2048-AES Homelab guide.
