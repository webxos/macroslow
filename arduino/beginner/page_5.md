# Project 5 of 10: DHT11 Temp/Humidity  
**Mastery Goal**: Digital Sensor Communication & Environmental Monitoring  

---

## Overview  
The DHT11 project teaches how to read **temperature** and **humidity** using a low-cost digital sensor. Key concepts:  
- One-wire protocol (timing-based).  
- Use **DHT library** for reliable decoding.  
- Parse °C, °F, and %RH.  
- Trigger alerts (e.g., LED if too hot).  
- Validate checksum for data integrity.  

Essential for drone **thermal safety**, **battery health**, or **weather-aware flight**.

---

## Components Needed  
| Item | Quantity | Notes |
|------|----------|-------|
| Arduino Uno | 1 | From kit |
| DHT11 Sensor | 1 | 3 or 4-pin module |
| 10kΩ Resistor | 1 | Pull-up for data line |
| LED + 220Ω Resistor | 1 | Alert (Pin 13) |
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
           │  DHT11  │
+5V ───────┤─●───●───┤
           │  │   │  │
          Pin2 │  NC │
               │     │
          ┌────┴─────┴─────┐
          │     10kΩ       │
         ┌┴──────┐         │
        GND      │         │
                 │         │
           ┌─────┴─────────┴────┐
           │         LED        │
Pin 13 ───┤─│220│───────────────┤──► Anode
           │ └───┘              │
           └──────────┬─────────┘
                      │
                     GND
```

**Step-by-Step Wiring**:  
1. **DHT11 (4-pin module)**:  
   - VCC → **+5V**  
   - DATA → **Pin 2**  
   - NC → leave open  
   - GND → **GND**  
2. **10kΩ Pull-up**: Between **+5V** and **DATA (Pin 2)**.  
3. **LED**: Pin 13 → 220Ω → anode → GND.  
4. **Power**: USB.

> **Note**: DHT11 needs **5–10kΩ pull-up**. Module versions often include it.

---

## Arduino Code (Full Sketch)  

```cpp
/*
  Project 5: DHT11 Temperature & Humidity
  Mastery: Digital Sensor & Alert Logic
  Hardware: DHT11 (Pin 2), LED (Pin 13)
*/

#include <DHT.h>

#define DHTPIN 2
#define DHTTYPE DHT11
#define LEDPIN 13

DHT dht(DHTPIN, DHTTYPE);

float humidity = 0.0;
float tempC = 0.0;
float tempF = 0.0;
const float HOT_THRESHOLD = 30.0;  // °C

void setup() {
  pinMode(LEDPIN, OUTPUT);
  Serial.begin(9600);
  dht.begin();
  Serial.println("DHT11 Sensor Ready");
  Serial.println("Hum(%)\tTemp(C)\tTemp(F)\tStatus\tLED");
  
  delay(2000);  // Sensor warm-up
}

void loop() {
  // Read with timeout
  humidity = dht.readHumidity();
  tempC = dht.readTemperature();
  tempF = dht.readTemperature(true);  // Fahrenheit

  // Check for read failure
  if (isnan(humidity) || isnan(tempC) || isnan(tempF)) {
    Serial.println("Failed to read from DHT11!");
    digitalWrite(LEDPIN, HIGH);  // Flash error
    delay(500);
    digitalWrite(LEDPIN, LOW);
    delay(1500);
    return;
  }

  // Alert: LED ON if too hot
  if (tempC > HOT_THRESHOLD) {
    digitalWrite(LEDPIN, HIGH);
  } else {
    digitalWrite(LEDPIN, LOW);
  }

  // Print every 2 seconds (DHT11 limit)
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint >= 2000) {
    Serial.print(humidity, 1);
    Serial.print("\t");
    Serial.print(tempC, 1);
    Serial.print("\t");
    Serial.print(tempF, 1);
    Serial.print("\t");
    Serial.print(tempC > HOT_THRESHOLD ? "HOT " : "OK  ");
    Serial.println(tempC > HOT_THRESHOLD ? "ON " : "OFF");
    lastPrint = millis();
  }

  delay(100);
}
```

---

## Upload & Test  

1. **Install Library**:  
   - Arduino IDE → **Sketch → Include Library → Manage Libraries**  
   - Search **"DHT sensor library" by Adafruit** → Install  
   - Also install **Adafruit Unified Sensor** if prompted  
2. Upload code.  
3. Open **Serial Monitor** (9600 baud).  
4. **Breathe on sensor** → humidity ↑, temp ↑ → LED ON if >30°C.  

**Expected Result**:  
- Normal room: `50.0%  24.5°C  76.1°F  OK  OFF`  
- Hot alert: `65.0%  32.1°C  89.8°F  HOT ON`  

---

## Troubleshooting  

| Issue | Cause | Fix |
|------|-------|-----|
| `Failed to read` | No pull-up | Add 10kΩ +5V→DATA |
| All zeros | Wrong pin | Use Pin 2 |
| Slow/no update | Delay <2s | DHT11 max 1 read/sec |
| Library error | Missing lib | Install DHT.h |

---

## Mastery Checkpoint ✅  
You have mastered DHT11 if:  
- [ ] Temp/humidity update **every ~2s**.  
- [ ] LED **ON only when >30°C**.  
- [ ] You can change threshold to **25°C** and test.  

**Next Challenge**: Add **heat index** calculation and print warning if >27°C (feels-like).  

---

**Proceed to Project 6 → MPU6050 Gyro/Accel** when ready.  
*All artifacts updated for Page 5.*
