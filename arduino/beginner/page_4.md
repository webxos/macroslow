# Project 4 of 10: LDR Light Sensor  
**Mastery Goal**: Analog Light Sensing & Threshold Control  

---

## Overview  
The LDR (Light-Dependent Resistor) project teaches how to measure ambient light intensity using a **voltage divider**. Key concepts:  
- LDR resistance drops in light, rises in dark.  
- Build divider: LDR + fixed resistor → analog voltage.  
- Use `analogRead()` to detect light levels.  
- Trigger actions (LED on in dark) with thresholds.  
- Auto-calibrate min/max for robustness.  

Vital for drone **auto-exposure**, **landing light**, or **night mode**.

---

## Components Needed  
| Item | Quantity | Notes |
|------|----------|-------|
| Arduino Uno | 1 | From kit |
| LDR (Photoresistor) | 1 | GL5528 or similar |
| 10kΩ Resistor | 1 | Pull-down |
| LED + 220Ω Resistor | 1 | From Project 1 |
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
           │   LDR   │
+5V ───────┤─●───────┤
           │         │
          A0   ┌─────┴─────┐
               │ 10kΩ      │
              ┌┴──────┐    │
             GND      │    │
                      │    │
           ┌──────────┴────┴────┐
           │        LED         │
Pin 13 ───┤─│220│───────────────┤──► Anode
           │ └───┘              │
           └──────────┬─────────┘
                      │
                     GND
```

**Step-by-Step Wiring**:  
1. **LDR**: One leg to **+5V**, other to **A0**.  
2. **10kΩ Resistor**: From **A0** to **GND** (pull-down).  
3. **LED**: Pin 13 → 220Ω → LED anode → GND.  
4. **Power**: USB.

> **Divider Formula**: `Vout = 5V × (10k / (10k + R_LDR))`  
> Bright → R_LDR low → Vout high (~5V)  
> Dark → R_LDR high → Vout low (~0V)

---

## Arduino Code (Full Sketch)  

```cpp
/*
  Project 4: LDR Light Sensor with Auto-Calibration
  Mastery: Light Threshold & Auto-Exposure
  Hardware: LDR+A0, 10k pull-down, LED on Pin 13
*/

const int ldrPin = A0;
const int ledPin = 13;

int lightValue = 0;
int minLight = 1023;   // Darkest seen
int maxLight = 0;      // Brightest seen
int threshold = 400;   // Auto LED trigger (adjustable)

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(9600);
  Serial.println("LDR Light Sensor Ready");
  Serial.println("Value\tNorm(%)\tLED");
  
  // Initial calibration delay
  delay(1000);
}

void loop() {
  lightValue = analogRead(ldrPin);
  
  // Auto-calibrate min/max over time
  if (lightValue < minLight) minLight = lightValue;
  if (lightValue > maxLight) maxLight = lightValue;
  
  // Normalize to 0–100%
  int normPercent = map(lightValue, minLight, maxLight, 0, 100);
  normPercent = constrain(normPercent, 0, 100);
  
  // LED on if too dark
  if (lightValue < threshold) {
    digitalWrite(ledPin, HIGH);
  } else {
    digitalWrite(ledPin, LOW);
  }
  
  // Print every 200ms
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 200) {
    Serial.print(lightValue);
    Serial.print("\t");
    Serial.print(normPercent);
    Serial.print("\t");
    Serial.println(lightValue < threshold ? "ON " : "OFF");
    lastPrint = millis();
  }
  
  delay(50);
}
```

---

## Upload & Test  

1. Upload code.  
2. Open **Serial Monitor** (9600 baud).  
3. **Cover LDR** → dark → LED **ON**.  
4. **Expose to light** (phone flashlight) → LED **OFF**.  

**Expected Result**:  
- Dark room: `~50–200` → LED ON  
- Bright: `~600–900` → LED OFF  
- Serial:  
  ```
  Value   Norm(%) LED
  120     15      ON
  750     88      OFF
  ```

---

## Troubleshooting  

| Issue | Cause | Fix |
|------|-------|-----|
| LED always ON | LDR wired backward | Swap legs |
| Values stuck | No pull-down | Add 10kΩ to GND |
| No change | Wrong analog pin | Use A0–A5 |
| Jitter | Noise | Average 4–8 readings |

---

## Mastery Checkpoint ✅  
You have mastered light sensing if:  
- [ ] LED turns **ON only in dark** (< threshold).  
- [ ] `normPercent` goes **0% (dark) → 100% (bright)**.  
- [ ] You can adjust `threshold = 300` and test.  

**Next Challenge**: Add **hysteresis** (ON at 300, OFF at 400) to prevent flicker.  

---

**Proceed to Project 5 → DHT11 Temp/Humidity** when ready.  
*All artifacts updated for Page 4.*
