# Project 2 of 10: Button Input  
**Mastery Goal**: Digital Input & Debouncing  

---

## Overview  
The Button Input project teaches how to read digital inputs from a pushbutton. Key concepts:  
- Configure pin as **INPUT_PULLUP** (internal resistor).  
- Detect **HIGH → LOW** transition on press.  
- Implement **software debouncing** to avoid false triggers.  
- Toggle LED state with each press (like drone arm/disarm).  

Essential for user controls, mode switches, or emergency stops on the bodycam drone.

---

## Components Needed  
| Item | Quantity | Notes |
|------|----------|-------|
| Arduino Uno | 1 | From kit |
| Pushbutton (tactile) | 1 | 4-pin |
| 10kΩ Resistor (optional) | 1 | For external pull-up |
| LED + 220Ω Resistor | 1 | From Project 1 |
| Breadboard & Jumpers | As needed | |

---

## Circuit Wiring (Breadboard Layout)  

```
Breadboard View:
+5V ───────●──────────────────┐
           │                  │
GND ───────┴────●─────────────┘
                │
           ┌────┴────┐
           │ Button  │
Pin 2 ───┤─●─────●───┤──► +5V
           │         │
           └────┬────┘
                │
               GND
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
1. **Button**: One pin to **Pin 2**, opposite pin to **+5V**.  
2. **Internal Pull-up**: No external resistor needed (uses `INPUT_PULLUP`).  
3. **LED**: Reuse from Project 1 → Pin 13 → 220Ω → LED → GND.  
4. **Ground**: Connect Arduino GND to breadboard GND rail.

> **Why Pull-up?** Prevents floating input when button open.

---

## Arduino Code (Full Sketch)  

```cpp
/*
  Project 2: Button Input with Debounce
  Mastery: Digital Input & State Toggle
  Hardware: Button (Pin 2) + LED (Pin 13)
*/

const int buttonPin = 2;     // Pushbutton pin
const int ledPin = 13;       // LED pin

int buttonState = 0;         // Current reading
int lastButtonState = HIGH;  // Previous (for debounce)
int ledState = LOW;          // LED toggle state

unsigned long lastDebounceTime = 0;
unsigned long debounceDelay = 50;  // ms

void setup() {
  pinMode(buttonPin, INPUT_PULLUP);  // Internal pull-up
  pinMode(ledPin, OUTPUT);
  digitalWrite(ledPin, ledState);
  
  Serial.begin(9600);
  Serial.println("Button Input Test Ready");
}

void loop() {
  int reading = digitalRead(buttonPin);  // LOW when pressed

  // Debounce logic
  if (reading != lastButtonState) {
    lastDebounceTime = millis();
  }
  
  if ((millis() - lastDebounceTime) > debounceDelay) {
    if (reading != buttonState) {
      buttonState = reading;
      
      // Toggle LED only on press (LOW)
      if (buttonState == LOW) {
        ledState = !ledState;
        digitalWrite(ledPin, ledState);
        Serial.print("Button Pressed - LED: ");
        Serial.println(ledState ? "ON" : "OFF");
      }
    }
  }
  
  lastButtonState = reading;
}
```

---

## Upload & Test  

1. Upload code.  
2. Open **Serial Monitor** (9600 baud).  
3. **Press button** → LED toggles ON/OFF.  

**Expected Result**:  
- LED starts OFF.  
- First press → ON.  
- Second press → OFF.  
- Serial prints on each press:  
  ```
  Button Pressed - LED: ON
  Button Pressed - LED: OFF
  ```

---

## Troubleshooting  

| Issue | Cause | Fix |
|------|-------|-----|
| LED toggles erratically | Bounce noise | Increase `debounceDelay` to 100 |
| Button does nothing | Wrong pin/wiring | Verify Pin 2 & pull-up |
| Always ON | No pull-up | Use `INPUT_PULLUP` or add 10kΩ |
| Serial spam | No debounce | Ensure debounce logic active |

---

## Mastery Checkpoint ✅  
You have mastered digital input if:  
- [ ] LED toggles **only once per press**.  
- [ ] You can change `debounceDelay` and observe effect.  
- [ ] You can move button to **Pin 3** and update code.  

**Next Challenge**: Add **double-press detection** (two presses in <500ms → flash LED 3x).  

---

**Proceed to Project 3 → Potentiometer ADC** when ready.  
*All artifacts updated for Page 2.*
