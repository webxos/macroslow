# 🏜️ MACROSLOW Home Lab Guide 2025: Page 7 – Edge & Medical IoT (Pi + BELUGA)

This page details the deployment and configuration of **Raspberry Pi edge clusters** with **BELUGA Agent 4.0** for real-time IoT, sensor fusion, and **HIPAA-compliant medical data pipelines** across all three **MACROSLOW 2048-AES builds**. Focus is on wearable integration (ECG, SpO2, temperature), **Kalman-filtered vitals fusion**, secure MQTT over TLS, and **PHI-encrypted transmission** to the DUNES gateway—enabling home health monitoring, clinic telemetry, or industrial sensor networks.

---

## 🛠️ Step 1: Raspberry Pi OS & Hardening (All Builds)

### Flash & Boot
```bash
# On host: Flash Ubuntu Server 24.04 ARM64 to microSD
sudo dd if=ubuntu-24.04-server-arm64.img of=/dev/mmcblk0 bs=4M status=progress && sync
```

### Initial Setup (Per Pi)

# Boot, login (ubuntu/ubuntu), change password
sudo passwd
sudo hostnamectl set-hostname pi1  # pi2, pi3, etc.

# Static IP (VLAN-aware)
sudo nano /etc/netplan/01-netcfg.yaml

```yaml
network:
  ethernets:
    eth0:
      dhcp4: no
      addresses: [192.168.1.101/24]  # pi1
      gateway4: 192.168.1.1
      nameservers:
        addresses: [8.8.8.8]
  vlans:
    vlan.100:
      id: 100
      link: eth0
      addresses: [172.16.100.101/24]  # Medical VLAN
```
```bash
sudo netplan apply
```

### Security Hardening
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install ufw fail2ban -y
``` 
# Firewall: Allow SSH, MQTT, DUNES
sudo ufw allow from 192.168.1.0/24 to any port 22
sudo ufw allow 1883,8883  # MQTT
sudo ufw enable

# SSH: Key-only
sudo sed -i 's/#PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart ssh


---

## 📡 Step 2: BELUGA Agent Installation

```bash
# Install BELUGA
curl -fsSL https://get.beluga.agent | sh
beluga version  # 4.0.1
```

### Generate TLS Certs (CA-Signed or Self-Signed)
```bash
# On DUNES server (CA)
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 -out ca.crt

# Per Pi: Client cert
openssl genrsa -out pi1.key 2048
openssl req -new -key pi1.key -out pi1.csr
openssl x509 -req -in pi1.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out pi1.crt -days 365 -sha256
```

---

## 🔧 Step 3: BELUGA Configuration Profiles

### Minimalist DUNES (Environmental Sensors)
```bash
beluga config set --role sensor
beluga config set --server https://dunes.local:8080
beluga config set --mqtt-broker mqtt://192.168.1.10:1883
beluga config set --interval 10s
```

### Chimera GH200 (Telemetry & GPU Monitoring)
```bash
beluga config set --role telemetry
beluga config set --gpu-metrics true
beluga config set --nvlink-stats true
beluga config set --mqtt-broker mqtts://10.0.20.10:8883 --tls-cert pi1.crt --tls-key pi1.key
```

### Glastonbury Medical (Wearables + PHI)
```bash
beluga config set --role medical
beluga config set --hipaa-mode true
beluga config set --phi-encrypt aes-256
beluga config set --devices ecg,spo2,temp
beluga config set --mqtt-broker mqtts://172.16.100.10:8883 --tls
beluga config set --redact name,mrn,dob
```

### Start Agent
```bash
beluga start --daemon
journalctl -u beluga -f  # Monitor logs
```

---

## 🩺 Step 4: Medical Device Integration (Glastonbury)

### Hardware Connections
- **ECG**: ADS1292R module → SPI (CE0)
- **SpO2**: MAX30102 → I2C (SDA/SCL)
- **Temperature**: MLX90614 → I2C
- **GPIO Pinout**:
  - SPI0: MOSI=10, MISO=9, SCLK=11, CE0=8
  - I2C1: SDA=3, SCL=5

### Driver & Collection Script (`collect_vitals.py`)
```python
import spidev, busio, board, adafruit_ads1x15, adafruit_max30102, mlx90614
import paho.mqtt.client as mqtt
import json, time, encryption

spi = spidev.SpiDev(); spi.open(0,0); spi.max_speed_hz = 500000
i2c = busio.I2C(board.SCL, board.SDA)
ads = adafruit_ads1x15.ADS1115(i2c)
max30102 = adafruit_max30102.MAX30102(i2c)
mlx = mlx90614.MLX90614(i2c)

client = mqtt.Client()
client.tls_set(ca_certs="ca.crt", certfile="pi1.crt", keyfile="pi1.key")
client.connect("172.16.100.10", 8883)

while True:
    ecg = ads.read_adc(0) * 4.096 / 32768
    hr, spo2 = max30102.read_sequential()
    temp = mlx.object_temperature
    payload = encryption.aes_encrypt(json.dumps({
        "hr": hr, "spo2": spo2, "temp": temp, "ecg": ecg
    }))
    client.publish("phi/vitals/pi1", payload)
    time.sleep(1)
```

### Register Script in BELUGA
```bash
beluga task add --name vitals --script collect_vitals.py --interval 1s
```

---

## ⚡ Step 5: Real-Time Vitals Fusion (DUNES Server)

### MAML Fusion Workflow (`vitals_fusion.maml`)

# Real-Time Vitals Fusion with Kalman Filter

```iot
agents: [pi1, pi2, pi3]
input: mqtts://172.16.100.10:8883/phi/vitals
decrypt: aes-256
```

```ai
model: kalman_vitals_fusion.py
input: decrypted_stream
output: /results/fused_vitals.json
rate: 1Hz
```

```medical
hipaa: true
redact: mrn
audit: /var/log/hipaa/vitals_fusion.log
retain: 90d
```

```quantum
optional: qml_anomaly_detection
qubits: 4
```


### Run Fusion
```bash
dunes maml run vitals_fusion.maml --stream
```

---

## 📊 Step 6: Dashboard & Alerts

### Grafana Panel (Glastonbury)
```bash
# Add MQTT datasource → topic: phi/vitals/+
# Panels: HR, SpO2, Temp, ECG waveform
# Alert: HR > 100 or SpO2 < 92 → email/sms
```

### DUNES Alert Rule
```bash
dunes alert add --name bradycardia --condition "hr < 50" --action sms:+15551234567
```

---

## ✅ Step 7: Validation & Compliance

# Check agent status
dunes iot list

# Verify encryption
beluga logs | grep "AES-256"

# HIPAA audit
cat /var/log/hipaa/vitals_fusion.log | grep "REDACTED"

# Stress test
beluga benchmark --duration 1h --rate 100msg/s

---

## 🔗 Next Steps
Proceed to **Page 8: Monitoring & Optimization** to set up Prometheus, Grafana, and performance tuning for quantum, AI, and medical workloads.

*From Legacy to GH200 — Build Your Future with MACROSLOW 2025!* 🏜️

**xAI Artifact Updated**: File `macroslow-homelab-guide-2025.md` updated with Page 7 edge and medical IoT setup using Pi + BELUGA Agent.
