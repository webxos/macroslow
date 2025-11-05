# Project 8 of 10: Servo Control  
**Mastery Goal**: PWM Servo Positioning & Smooth Motion  

---

## Overview  
The Servo Control project teaches **precise angular control** using PWM signals. Key concepts:  
- Servo expects **1–2ms pulse every 20ms** (0–180°).  
- Use **Servo library** to generate signals.  
- Map analog input (pot) → servo angle.  
- Implement **smooth sweeping** with incremental steps.  

Essential for **drone gimbal**, **camera tilt**, and **landing gear**.

---

## Components Needed  
| Item | Quantity | Notes |
|------|----------|-------|
| Arduino Uno | 1 | From kit |
| SG90 Micro Servo | 1 | 180° rotation |
| 10kΩ Potentiometer | 1 | From Project 3 |
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
           │  Servo  │
+5V ───────┤─●───●───┤──► Red (Power)
           │  │   │  │
          Pin9 │  │  │
               │  │  │
              GND │  │
                  │  │
           ┌──────┴──┴────┐
           │     POT      │
+5V ───────┤─●───●───────┤──► GND
           │   │         │
          A0   │
               │
```

**Step-by-Step Wiring**:  
1. **Servo**:  
   - Red → **+5V**  
   - Brown/Black → **GND**  
   - Orange/Yellow → **Pin 9**  
2. **Potentiometer**:  
   - Left → **+5V**  
   - Right → **GND**  
   - Middle → **A0**  
3. **Power**: USB (servo draws ~100mA).

> **Warning**: For multiple servos, use **external 5V supply** with common GND.

---

## Arduino Code (Full Sketch)  

```cpp
/*
  Project 8: Servo Control with Potentiometer
  Mastery: PWM Positioning & Smooth Motion
  Hardware: Servo (Pin 9), Pot (A0)
*/

#include <Servo.h>

Servo myservo;
const int potPin = A0;
const int servoPin = 9;

int potValue = 0;
int targetAngle = 0;
int currentAngle = 0;
const int SMOOTH_STEP = 1;  // degrees per loop

void setup() {
  myservo.attach(servoPin);
  Serial.begin(9600);
  Serial.println("Servo Control Ready");
  Serial.println("Pot\tTarget\tCurrent");
  
  myservo.write(90);  // Center start
  delay(500);
}

void loop() {
  // Read pot (0–1023)
  potValue = analogRead(potPin);
  
  // Map to 0–180°
  targetAngle = map(potValue, 0, 1023, 0, 180);
  
  // Smooth motion: step toward target
  if (currentAngle < targetAngle) {
    currentAngle += SMOOTH_STEP;
    if (currentAngle > targetAngle) currentAngle = targetAngle;
  } else if (currentAngle > targetAngle) {
    currentAngle -= SMOOTH_STEP;
    if (currentAngle < targetAngle) currentAngle = targetAngle;
  }
  
  myservo.write(currentAngle);
  
  // Print every 100ms
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 100) {
    Serial.print(potValue);
    Serial.print("\t");
    Serial.print(targetAngle);
    Serial.print("\t");
    Serial.println(currentAngle);
    lastPrint = millis();
  }
  
  delay(15);  // ~60Hz update
}
```

---

## Upload & Test  

1. **Install Library**:  
   - **Servo** (built-in)  
2. Upload code.  
3. Open **Serial Monitor** (9600 baud).  
4. **Rotate pot slowly**:  
   - 0° → 180° → servo follows **smoothly**.  

**Expected Result**:  
```
Pot     Target  Current
23      4       4
512     90      88
1010    177     177
```

---

## Troubleshooting  

| Issue | Cause | Fix |
|------|-------|-----|
| Servo jitters | Power drop | Use external 5V |
| No movement | Signal wire | Check Pin 9 |
| Only 0–90° | Pot reversed | Swap +5V/GND |
| Stuttering | Delay too low | Increase to 20ms |

---

## Mastery Checkpoint ✅  
You have mastered servo control if:  
- [ ] Servo moves **smoothly** from 0° to 180°.  
- [ ] No jitter or sudden jumps.  
- [ ] You can change `SMOOTH_STEP = 3` and observe faster motion.  

**Next Challenge**: Add **button** to toggle between **manual pot** and **auto sweep** (0→180→0).  

---

**Proceed to Project 9 → ESP32 BLE Beacon** when ready.  
*All artifacts updated for Page 8.*
