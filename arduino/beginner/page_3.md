# Project 3 of 10: Potentiometer ADC  
**Mastery Goal**: Analog Input & Value Mapping  

---

## Overview  
The Potentiometer ADC project teaches how to read analog voltages (0–5V) from a potentiometer using the Arduino’s **10-bit ADC** (0–1023 range). Key concepts:  
- Use `analogRead()` on analog pins.  
- Map raw values to meaningful ranges (e.g., 0–180° for servos).  
- Smooth noisy readings with averaging.  
- Control LED brightness with PWM (analogWrite).  

Critical for drone throttle, gimbal control, or sensor scaling.

---

## Components Needed  
| Item | Quantity | Notes |
|------|----------|-------|
| Arduino Uno | 1 | From kit |
| 10kΩ Potentiometer | 1 | Linear taper |
| LED + 220Ω Resistor | 1 | From Project 1 |
| Breadboard & Jumpers | As needed | |

---

## Circuit Wiring (Breadboard Layout)  

```
Breadboard View:
+5V ───────●──────┐
           │      │
GND ───────┴──────┘
                  │
             ┌────┴────┐
             │  POT    │
+5V ────────┤─●───●───┤──► GND
             │   │    │
            A0   │
                 │
           ┌─────┴─────┐
           │    LED    │
Pin 9 ────┤─│220│──────┤──► Anode
           │ └───┘      │
           └─────┬──────┘
                 │
                GND
```

**Step-by-Step Wiring**:  
1. **Potentiometer**:  
   - Left pin → **+5V**  
   - Right pin → **GND**  
   - Middle (wiper) → **A0**  
2. **LED (PWM)**: Pin 9 → 220Ω → LED anode → GND.  
3. **Power**: USB or 9V.

> **Note**: Only pins with `~` support `analogWrite()` (3,5,6,9,10,11 on Uno).

---

## Arduino Code (Full Sketch)  

```cpp
/*
  Project 3: Potentiometer ADC with PWM
  Mastery: Analog Input & Mapping
  Hardware: 10k Pot (A0) → LED PWM (Pin 9)
*/

const int potPin = A0;      // Potentiometer wiper
const int ledPin = 9;       // PWM LED

int rawValue = 0;           // 0–1023
int brightness = 0;         // 0–255 (PWM)
int smoothed = 0;           // Averaged reading

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
  Serial.println("Potentiometer ADC Test Ready");
  Serial.println("Raw\tMapped\tBrightness");
}

void loop() {
  // Read & smooth (simple moving average)
  rawValue = analogRead(potPin);
  smoothed = (smoothed * 3 + rawValue) / 4;  // Reduce jitter
  
  // Map 0–1023 → 0–255 for PWM
  brightness = map(smoothed, 0, 1023, 0, 255);
  
  analogWrite(ledPin, brightness);  // Set LED brightness
  
  // Print every 100ms
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 100) {
    Serial.print(smoothed);
    Serial.print("\t");
    Serial.print(map(smoothed, 0, 1023, 0, 180));  // e.g., degrees
    Serial.print("\t");
    Serial.println(brightness);
    lastPrint = millis();
  }
  
  delay(10);  // Small delay for stability
}
```

---

## Upload & Test  

1. Upload code.  
2. Open **Serial Monitor** (9600 baud).  
3. **Rotate pot knob** slowly.  

**Expected Result**:  
- LED brightness changes smoothly from **OFF → FULL**.  
- Serial prints:  
  ```
  Raw     Mapped  Brightness
  12      2       3
  512     90      127
  1010    177     251
  ```

---

## Troubleshooting  

| Issue | Cause | Fix |
|------|-------|-----|
| LED stuck ON/OFF | Wrong pin | Use PWM pin (9) |
| Jumpy values | Noise | Increase smoothing or add 100nF cap across pot |
| No change | Pot wired backward | Swap +5V/GND |
| Values stuck at 0/1023 | Wiper not on A0 | Verify middle pin to A0 |

---

## Mastery Checkpoint ✅  
You have mastered analog input if:  
- [ ] LED brightness follows pot **smoothly**.  
- [ ] You can map to **0–180** and print as "degrees".  
- [ ] You can move pot to **A1** and LED to **Pin 10**.  

**Next Challenge**: Use pot to control **servo angle** (0–180°) instead of LED.  

---

**Proceed to Project 4 → LDR Light Sensor** when ready.  
*All artifacts updated for Page 3.*
