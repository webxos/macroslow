## Page 3: Programming Drone Flight Controllers
The **MACROSLOW SDK** transforms Arduino-based drones into **quantum-aware, self-learning, and swarm-capable systems** by building on robust flight controller firmware. This page provides **in-depth programming guidance** for Arduino IDE 2.x, covering **PID stabilization**, **ESC calibration**, **IMU fusion**, **failsafe logic**, and **MACROSLOW integration hooks** for **DUNES**, **CHIMERA**, and **GLASTONBURY**. All code is compatible with **Portenta H7**, **GIGA R1 WiFi**, **MKR WiFi 1010**, and **Nano**, with **MicroPython** and **C++** examples. Drawing from **MultiWii**, **Betaflight**, **ArduCopter**, and **Cleanflight**, this guide evolves traditional 6-DOF control into **hybrid classical-quantum pipelines** with **real-time telemetry**, **MAML logging**, and **API command injection**.

---

### Core Flight Controller Architecture
```
[Radio Input] → [RX Parsing] → [Setpoint] → [PID Loop] → [Motor Mix] → [ESC PWM]
      ↑              ↑             ↑            ↑           ↑
   Failsafe       IMU Fusion    QNN Bias     VQE Opt     QKD Secure
```

- **Setpoint**: Desired roll/pitch/yaw/throttle from radio or MCP.
- **IMU Fusion**: MPU6050 + Madgwick/Mahony filter for attitude.
- **PID Loop**: Three cascaded loops (rate → angle → output).
- **Motor Mix**: Quad X configuration mapping.
- **ESC Output**: 1000–2000μs PWM (OneShot125, DShot optional).

---

### ESC Calibration & Motor Spin Test
```cpp
// ESC Calibration Sketch (Run once with battery disconnected from ESCs)
#include <Servo.h>
Servo esc1, esc2, esc3, esc4;

void setup() {
  esc1.attach(3); esc2.attach(5); esc3.attach(6); esc4.attach(9);
  Serial.begin(115200);
  Serial.println("Connect battery. Wait for high beep.");
  delay(5000);
  writeESC(2000); // Max throttle
  Serial.println("High beep heard? Connect battery now.");
  delay(7000);
  writeESC(1000); // Min throttle
  Serial.println("Low beep. Calibration complete.");
}

void writeESC(int val) {
  esc1.writeMicroseconds(val); esc2.writeMicroseconds(val);
  esc3.writeMicroseconds(val); esc4.writeMicroseconds(val);
}

void loop() {}
```
> **Safety**: Remove props during calibration.

---

### MPU6050 Setup & Madgwick Filter
```cpp
#include <Wire.h>
#include <MPU6050_6Axis_MotionApps20.h>
MPU6050 mpu;
bool dmpReady = false;
uint16_t packetSize;
Quaternion q;
VectorFloat gravity;
float ypr[3]; // [yaw, pitch, roll]

void setup() {
  Wire.begin();
  mpu.initialize();
  mpu.dmpInitialize();
  mpu.setDMPEnabled(true);
  packetSize = mpu.dmpGetFIFOPacketSize();
}

void loop() {
  if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetGravity(&gravity, &q);
    mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
    // ypr[0] = yaw, ypr[1] = pitch, ypr[2] = roll (radians)
  }
}
```

---

### Full PID Flight Controller (C++ for Portenta H7 / GIGA R1)
```cpp
// MACROSLOW-Ready Flight Controller
#include <Wire.h>
#include <MPU6050_6Axis_MotionApps20.h>
#include <Servo.h>
#include <WiFiNINA.h> // or WiFi.h for GIGA

MPU6050 mpu;
Servo esc[4];
int escPin[4] = {3, 5, 6, 9};

// PID Gains (Tune with Ziegler-Nichols)
float Kp_rate = 0.6, Ki_rate = 0.8, Kd_rate = 0.05;
float Kp_angle = 4.0, Ki_angle = 0.0, Kd_angle = 0.1;

float setpoint[3] = {0, 0, 0}; // roll, pitch, yaw
float error[3], integral[3], derivative[3], lastError[3];
float output[3]; // rate PID output
float motor[4];
unsigned long lastTime;
bool armed = false;

// MACROSLOW Hooks
String mamlLog = "";
float qnnBias[3] = {0,0,0}; // From DUNES QNN

void setup() {
  Serial.begin(115200);
  WiFi.begin("DroneNet", "QKD_KEY"); // CHIMERA secured
  for(int i=0; i<4; i++) esc[i].attach(escPin[i]);
  mpu.initialize(); mpu.dmpInitialize(); mpu.setDMPEnabled(true);
  armESCs();
  lastTime = micros();
}

void loop() {
  // 1. Read IMU
  if (mpu.dmpGetCurrentFIFOPacket(fifoBuffer)) {
    mpu.dmpGetQuaternion(&q, fifoBuffer);
    mpu.dmpGetYawPitchRoll(ypr, &q, &gravity);
  }

  // 2. Read Radio / MCP Setpoint
  parseRadio(); // or parseMCP()

  // 3. Angle PID (Outer Loop)
  for(int i=1; i<3; i++) { // pitch, roll
    error[i] = setpoint[i] - ypr[i];
    integral[i] += error[i];
    derivative[i] = error[i] - lastError[i];
    output[i] = Kp_angle*error[i] + Ki_angle*integral[i] + Kd_angle*derivative[i];
    lastError[i] = error[i];
  }

  // 4. Rate PID (Inner Loop) + QNN Bias
  float dt = (micros() - lastTime) / 1e6;
  lastTime = micros();
  for(int i=0; i<3; i++) {
    float rate = (i<2) ? (ypr[i] - lastYPR[i])/dt : gyro[i];
    error[i] = (i<2) ? output[i] : setpoint[i] - rate;
    error[i] += qnnBias[i]; // DUNES QNN correction
    integral[i] += error[i]*dt;
    derivative[i] = (error[i] - lastError[i])/dt;
    motorOutput[i] = Kp_rate*error[i] + Ki_rate*integral[i] + Kd_rate*derivative[i];
    lastError[i] = error[i]; lastYPR[i] = ypr[i];
  }

  // 5. Motor Mixing (Quad X)
  motor[0] = throttle + motorOutput[1] + motorOutput[2] - motorOutput[0]; // FR
  motor[1] = throttle - motorOutput[1] + motorOutput[2] + motorOutput[0]; // FL
  motor[2] = throttle - motorOutput[1] - motorOutput[2] - motorOutput[0]; // RL
  motor[3] = throttle + motorOutput[1] - motorOutput[2] + motorOutput[0]; // RR

  // 6. Constrain & Write
  for(int i=0; i<4; i++) {
    motor[i] = constrain(motor[i], 1100, 1900);
    esc[i].writeMicroseconds(armed ? motor[i] : 1000);
  }

  // 7. MAML Telemetry Log (DUNES)
  logMAML();

  // 8. CHIMERA API Check
  checkAPICommands();
}
```

---

### MicroPython Version (Portenta H7)
```python
# flight_controller.py
import machine, time, network
from mpu6050 import MPU6050
from servo import Servo

i2c = machine.I2C(0)
mpu = MPU6050(i2c)
esc = [Servo(machine.Pin(p)) for p in [3,5,6,9]]

def pid_angle(error, kp=4.0):
    return kp * error

while True:
    angles = mpu.get_angles()
    error_pitch = setpoint_pitch - angles['pitch']
    output = pid_angle(error_pitch) + qnn_bias  # From DUNES
    # Motor mix and write
    time.sleep(0.01)
```

---

### MACROSLOW Integration Hooks
| **Hook** | **SDK** | **Function** |
|---------|--------|-------------|
| `qnnBias[3]` | **DUNES** | Inject QNN correction into PID |
| `checkAPICommands()` | **CHIMERA** | Receive QKD-secured MCP commands |
| `logMAML()` | **DUNES** | Write `.maml.md` with 256-bit AES |
| `parseMCP()` | **GLASTONBURY** | Receive VQE-optimized setpoints |

---

### Failsafe & Arming Logic
```cpp
void failsafe() {
  if (radioLost() || batteryLow()) {
    armed = false;
    for(int i=0; i<4; i++) esc[i].writeMicroseconds(1000);
    logMAML("FAILSAFE");
  }
}
```

---

### Tuning Guide
1. **Rate PID**: Start with `Kp=0.6, Ki=0.8, Kd=0.05`. Increase Kp until oscillation.
2. **Angle PID**: `Kp=4.0` for stable hover.
3. **QNN Bias**: Start at 0, let DUNES train mid-flight.

---

### Community & Resources
- **MultiWii 2.3**: [github.com/multiwii](https://github.com/multiwii) – Base PID.
- **Betaflight Configurator**: Tuning reference.
- **Arduino Drone Forum**: [forum.arduino.cc/t/drone](https://forum.arduino.cc) – ESC timing, IMU drift.
- **YMFC-AL**: YouTube auto-level tutorials.

**Next**: Page 4 introduces **QNN for Drone Autonomy** with PyTorch/Qiskit hybrid models.

**License**: © 2025 WebXOS Research Group. MIT License; fork at [github.com/webxos/macroslow](https://github.com/webxos/macroslow).