# Project 6 of 10: MPU6050 Gyro/Accel  
**Mastery Goal**: I²C Communication & Attitude Sensing  

---

## Overview  
The MPU6050 project teaches **6-DOF motion tracking** using a 3-axis accelerometer and 3-axis gyroscope. Key concepts:  
- **I²C protocol** (SDA/SCL).  
- Read raw sensor data via **Adafruit MPU6050 library**.  
- Convert to **pitch/roll angles** using complementary filter.  
- Detect **tilt → LED alert**.  

Critical for **drone stabilization**, **gimbal lock**, and **crash detection**.

---

## Components Needed  
| Item | Quantity | Notes |
|------|----------|-------|
| Arduino Uno | 1 | From kit |
| MPU6050 Module | 1 | GY-521 breakout |
| LED + 220Ω Resistor | 1 | Tilt alert |
| Breadboard & Jumpers | As needed | |

---

## Circuit Wiring (Breadboard Layout)  

```
Breadboard View:
+5V ───────●────────────────┐
           │                │
GND ───────┴────●───────────┘
                │
           ┌────┴────┐
           │ MPU6050 │
+5V ───────┤─●───●───┤
           │  │   │  │
          A4  A5  NC │
           │  │      │
           └─────────┘
                │
           ┌────┴────┐
           │  LED    │
Pin 13 ───┤─│220│───┤──► Anode
           │ └───┘   │
           └────┬────┘
                │
               GND
```

**Step-by-Step Wiring**:  
1. **MPU6050**:  
   - VCC → **+5V**  
   - GND → **GND**  
   - SCL → **A5** (Uno)  
   - SDA → **A4** (Uno)  
2. **LED**: Pin 13 → 220Ω → anode → GND.  
3. **Power**: USB (MPU6050 has 3.3V regulator).

> **I²C Address**: 0x68 (default). AD0 = GND.

---

## Arduino Code (Full Sketch)  

```cpp
/*
  Project 6: MPU6050 Gyro/Accel with Tilt Detection
  Mastery: I2C & Complementary Filter
  Hardware: MPU6050 (A4/A5), LED (Pin 13)
*/

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;
const int ledPin = 13;
const float TILT_THRESHOLD = 45.0;  // degrees

// Complementary filter
float pitch = 0.0, roll = 0.0;
float alpha = 0.98;  // Gyro weight

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
  Wire.begin();
  
  if (!mpu.begin()) {
    Serial.println("MPU6050 not found!");
    while (1) delay(10);
  }
  
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
  
  Serial.println("MPU6050 Ready");
  Serial.println("Pitch\tRoll\tTilt\tLED");
  delay(100);
}

void loop() {
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);
  
  // Accel angles (degrees)
  float accelPitch = atan2(a.acceleration.y, a.acceleration.z) * 180 / PI;
  float accelRoll  = atan2(-a.acceleration.x, sqrt(a.acceleration.y * a.acceleration.y + a.acceleration.z * a.acceleration.z)) * 180 / PI;
  
  // Gyro integration (dt in seconds)
  static unsigned long lastTime = 0;
  float dt = (millis() - lastTime) / 1000.0;
  lastTime = millis();
  
  pitch = alpha * (pitch + g.gyro.x * dt * 180 / PI) + (1 - alpha) * accelPitch;
  roll  = alpha * (roll  + g.gyro.y * dt * 180 / PI) + (1 - alpha) * accelRoll;
  
  // Tilt alert
  bool tilted = (abs(pitch) > TILT_THRESHOLD || abs(roll) > TILT_THRESHOLD);
  digitalWrite(ledPin, tilted ? HIGH : LOW);
  
  // Print every 100ms
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 100) {
    Serial.print(pitch, 1);
    Serial.print("\t");
    Serial.print(roll, 1);
    Serial.print("\t");
    Serial.print(tilted ? "YES" : "NO ");
    Serial.println(tilted ? "\tON " : "\tOFF");
    lastPrint = millis();
  }
  
  delay(10);
}
```

---

## Upload & Test  

1. **Install Libraries**:  
   - **Adafruit MPU6050**  
   - **Adafruit Unified Sensor**  
   - **Wire** (built-in)  
2. Upload code.  
3. Open **Serial Monitor** (9600 baud).  
4. **Tilt board**:  
   - Forward → pitch ↑  
   - Side → roll ↑  
   - >45° → LED **ON**  

**Expected Result**:  
```
Pitch   Roll    Tilt    LED
-2.1    1.3     NO      OFF
48.7    5.2     YES     ON
```

---

## Troubleshooting  

| Issue | Cause | Fix |
|------|-------|-----|
| `not found` | I²C wiring | Check A4/A5, pull-ups |
| Stuck at 0 | No motion | MPU flat on table |
| Drifting angles | Filter off | Adjust `alpha` |
| Noisy | Vibration | Add 100nF caps on VCC/GND |

---

## Mastery Checkpoint ✅  
You have mastered MPU6050 if:  
- [ ] Pitch/roll update **smoothly** with tilt.  
- [ ] LED **ON only when >45°**.  
- [ ] You can change `TILT_THRESHOLD = 30` and test.  

**Next Challenge**: Add **yaw integration** using magnetometer (if HMC5883L available).  

---

**Proceed to Project 7 → HC-SR04 Ultrasonic** when ready.  
*All artifacts updated for Page 6.*
