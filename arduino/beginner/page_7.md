# Project 7 of 10: HC-SR04 Ultrasonic  
**Mastery Goal**: Distance Sensing & Pulse Timing  

---

## Overview  
The HC-SR04 project teaches **ultrasonic distance measurement** using sound wave reflection. Key concepts:  
- Trigger 10µs pulse → sensor emits 40kHz burst.  
- Measure **ECHO pulse width** → time-of-flight.  
- Distance = `(duration × 0.0343) / 2` cm (speed of sound 343 m/s).  
- Filter noise with median averaging.  
- Alert (LED) if object < 20cm.  

Critical for drone **collision avoidance**, **precision landing**, and **proximity detection**.

---

## Components Needed  
| Item | Quantity | Notes |
|------|----------|-------|
| Arduino Uno | 1 | From kit |
| HC-SR04 Sensor | 1 | 4-pin module |
| LED + 220Ω Resistor | 1 | Alert |
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
           │ HC-SR04 │
+5V ───────┤─●───●───┤
           │  │   │  │
         Pin9 Pin8   │
           │         │
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
1. **HC-SR04**:  
   - VCC → **+5V**  
   - GND → **GND**  
   - TRIG → **Pin 9**  
   - ECHO → **Pin 8**  
2. **LED**: Pin 13 → 220Ω → anode → GND.  
3. **Power**: USB.

> **Note**: Sensor works 2–400cm. Avoid soft surfaces (sound absorption).

---

## Arduino Code (Full Sketch)  

```cpp
/*
  Project 7: HC-SR04 Ultrasonic Distance Sensor
  Mastery: Pulse Timing & Median Filter
  Hardware: TRIG Pin 9, ECHO Pin 8, LED Pin 13
*/

const int trigPin = 9;
const int echoPin = 8;
const int ledPin = 13;
const int PROXIMITY_THRESHOLD = 20;  // cm

float distance = 0.0;

void setup() {
  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
  Serial.println("HC-SR04 Ultrasonic Ready");
  Serial.println("Distance(cm)\tStatus\tLED");
}

float readDistance() {
  // Trigger pulse
  digitalWrite(trigPin, LOW); delayMicroseconds(2);
  digitalWrite(trigPin, HIGH); delayMicroseconds(10);
  digitalWrite(trigPin, LOW);
  
  // Measure echo
  long duration = pulseIn(echoPin, HIGH, 30000);  // timeout 30ms (~5m)
  
  if (duration == 0) return -1;  // timeout
  
  // Speed of sound: 0.0343 cm/µs
  return (duration * 0.0343) / 2.0;
}

void loop() {
  // Take 5 readings, sort, take median
  float readings[5];
  for (int i = 0; i < 5; i++) {
    readings[i] = readDistance();
    delay(50);
  }
  
  // Simple bubble sort
  for (int i = 0; i < 5; i++) {
    for (int j = i + 1; j < 5; j++) {
      if (readings[i] > readings[j]) {
        float temp = readings[i];
        readings[i] = readings[j];
        readings[j] = temp;
      }
    }
  }
  
  distance = readings[2];  // Median
  
  // Alert
  bool tooClose = (distance > 0 && distance < PROXIMITY_THRESHOLD);
  digitalWrite(ledPin, tooClose ? HIGH : LOW);
  
  // Print every 200ms
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 200) {
    if (distance < 0) {
      Serial.println("ERROR\t-\tOFF");
    } else {
      Serial.print(distance, 1);
      Serial.print("\t");
      Serial.print(tooClose ? "CLOSE" : "SAFE ");
      Serial.println(tooClose ? "\tON " : "\tOFF");
    }
    lastPrint = millis();
  }
  
  delay(50);
}
```

---

## Upload & Test  

1. Upload code.  
2. Open **Serial Monitor** (9600 baud).  
3. **Move hand/object**:  
   - >20cm → LED **OFF**  
   - <20cm → LED **ON**  

**Expected Result**:  
```
Distance(cm)    Status  LED
45.2            SAFE    OFF
18.7            CLOSE   ON
```

---

## Troubleshooting  

| Issue | Cause | Fix |
|------|-------|-----|
| Always 0 | No echo | Check wiring, power |
| Jumpy values | Noise | Use median filter |
| Max 400cm | Out of range | Move closer |
| No pulse | TRIG/ECHO swapped | Verify Pin 9/8 |

---

## Mastery Checkpoint ✅  
You have mastered ultrasonic sensing if:  
- [ ] Distance updates **smoothly** 2–200cm.  
- [ ] LED **ON only <20cm**.  
- [ ] You can change threshold to **10cm** and test.  

**Next Challenge**: Add **moving average** and detect **approaching object** (blink LED).  

---

**Proceed to Project 8 → Servo Control** when ready.  
*All artifacts updated for Page 7.*
