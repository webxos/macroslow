# Project 1 of 10: LED Blink Test  
**Mastery Goal**: Basic Digital Output Control  

---

## Overview  
The LED Blink Test is the foundational project for understanding how to control digital outputs on the Arduino. By connecting an LED to a digital pin and programming the Arduino to turn it on and off at regular intervals, you will learn:  
- How to configure a pin as an output.  
- How to set a digital pin HIGH (5V) or LOW (0V).  
- The structure of `setup()` and `loop()` functions.  
- Basic timing using `delay()`.  

This skill is essential for controlling motors, relays, LEDs, and status indicators on your bodycam drone.

---

## Components Needed  
| Item | Quantity | Notes |
|------|----------|-------|
| Arduino Uno (or compatible) | 1 | From your Ultimate Kit |
| 5mm LED (any color) | 1 | Red, green, or yellow recommended |
| 220Ω Resistor | 1 | Protects LED from overcurrent |
| Breadboard | 1 | For prototyping |
| Jumper Wires | 2 | Male-to-male |

---

## Circuit Wiring (Breadboard Layout)  

```
Breadboard View:
+5V Rail ────────────────────────
                                  │
GND Rail ──────●──────────────────┘
               │
          ┌────┴────┐
          │  LED    │
          │  ┌───┐  │
Pin 13 ───┤─│220│──┤──► LED Anode (+)
          │ └───┘  │
          └────┬────┘
               │
              GND
```

**Step-by-Step Wiring Instructions**:  
1. **Insert LED**: Place LED on breadboard. **Longer leg (anode)** faces Pin 13 side.  
2. **Add Resistor**: Connect 220Ω resistor from **Pin 13** to **LED anode**.  
3. **Connect Cathode**: Short leg (cathode) to **GND rail**.  
4. **Ground Arduino**: Connect Arduino GND to breadboard GND rail.  
5. **Power (Optional)**: For standalone, power via USB or 9V battery.

> **Pro Tip**: Always use a current-limiting resistor (220Ω–330Ω) with LEDs to prevent burnout.

---

## Arduino Code (Full Sketch)  

```cpp
/*
  Project 1: LED Blink Test
  Mastery: Digital Output Control
  Hardware: Arduino + LED + 220Ω Resistor
*/

void setup() {
  // Initialize digital pin 13 as an OUTPUT
  pinMode(13, OUTPUT);
  
  // Optional: Initialize Serial for debugging
  Serial.begin(9600);
  Serial.println("LED Blink Test Started");
}

void loop() {
  digitalWrite(13, HIGH);   // Turn LED ON (5V)
  Serial.println("LED ON");
  delay(1000);              // Wait 1 second (1000 ms)
  
  digitalWrite(13, LOW);    // Turn LED OFF (0V)
  Serial.println("LED OFF");
  delay(1000);              // Wait 1 second
}
```

---

## Upload & Test  

1. Open **Arduino IDE**.  
2. Connect Arduino via USB.  
3. Select **Board**: `Arduino Uno` (Tools → Board).  
4. Select **Port**: Your COM port (Tools → Port).  
5. Paste code → Click **Upload**.  
6. Open **Serial Monitor** (Ctrl+Shift+M) → Set baud to **9600**.  

**Expected Result**:  
- LED blinks **ON for 1 second**, **OFF for 1 second**.  
- Serial Monitor prints:  
  ```
  LED Blink Test Started
  LED ON
  LED OFF
  LED ON
  ...
  ```

---

## Troubleshooting  

| Issue | Cause | Fix |
|------|-------|-----|
| LED not lighting | Wrong pin/resistor | Double-check Pin 13 & resistor |
| LED always ON | No `digitalWrite(LOW)` | Ensure both HIGH/LOW in loop |
| Faint LED | Wrong resistor value | Use 220Ω (not 10kΩ) |
| No Serial output | Baud mismatch | Set Serial Monitor to 9600 |

---

## Mastery Checkpoint ✅  
You have mastered digital output if:  
- [ ] LED blinks exactly every 1 second.  
- [ ] You can change blink speed by editing `delay()`.  
- [ ] You can move LED to another pin (e.g., Pin 12) and update code.  

**Next Challenge**: Modify code to blink **SOS in Morse code** (··· −−− ···).  

---

**Proceed to Project 2 → Button Input** when ready.  
*All artifacts updated for Page 1.*
