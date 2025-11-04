## 🐪 Micro Quantum Sim PC Breadboard Build

# *Arduino Uno + ESP32*  

**Version 1.0.0** | **WebXOS Research Group** | **Nov 04, 2025**  

**Repo:** github.com/webxos/dunes-qubit-sdk  
**Email:** project_dunes@outlook.com  

---

## PAGE 1: Get Your Parts & Build the Board  

### What You Need  
| Item | Qty | Notes |  
|------|-----|-------|  
| Arduino Uno R3 | 1 | Main brain |  
| ESP32 DevKit V1 | 1 | WiFi + math |  
| Breadboard (830 pt) | 1 | No soldering |  
| Jumper wires (M-M) | 20+ | Short ones |  
| LEDs (any color) | 4 | Qubit lights |  
| 220 Ω resistors | 4 | For LEDs |  
| 10 kΩ potentiometer | 1 | Twist for phase |  
| Buzzer (active) | 1 | Beep on error |  
| 5 V power (USB or 9 V battery) | 1 | Keep it steady |  

**Total cost:** ~$20  

---

### Step-by-Step Wiring  

**1. Power Rails**  
- Left rail: **5 V** (red) + **GND** (blue)  
- Right rail: **3.3 V** (red) + **GND** (blue) – for ESP32 only  

**2. Arduino Uno**  
| Arduino Pin | Connect To |  
|-------------|------------|  
| 5 V | Breadboard 5 V rail |  
| GND | Breadboard GND rail |  
| Pin 2 | LED1 anode → 220 Ω → GND |  
| Pin 3 | LED2 anode → 220 Ω → GND |  
| Pin 4 | LED3 anode → 220 Ω → GND |  
| Pin 5 | LED4 anode → 220 Ω → GND |  
| A0 | Pot middle pin |  
| A1 | Pot left pin → GND |  
| A2 | Pot right pin → 5 V |  
| Pin 13 | Built-in LED (heartbeat) |  

**3. ESP32**  
| ESP32 Pin | Connect To |  
|-----------|------------|  
| 3V3 | Breadboard 3.3 V rail |  
| GND | Breadboard GND rail |  
| GPIO 21 (SDA) | Arduino A4 |  
| GPIO 22 (SCL) | Arduino A5 |  
| GPIO 2 | Built-in LED (gate blink) |  
| GPIO 16 (TX) | Arduino Pin 8 (RX) |  
| GPIO 17 (RX) | Arduino Pin 9 (TX) |  

**4. Buzzer**  
- + leg → Arduino Pin 12  
- – leg → GND  

**5. Quick Check**  
- No smoke?  
- All LEDs light when you touch 5 V to cathode?  
- Pot turns smoothly?  

**Done in 15 min.**  

---

### Flash the Brains  

**Arduino (IDE)**  
1. Open Arduino IDE  
2. Paste this (Page 2 has full code):  

```cpp
#include <Wire.h>
void setup() {
  pinMode(2, OUTPUT); pinMode(3, OUTPUT);
  pinMode(4, OUTPUT); pinMode(5, OUTPUT);
  pinMode(13, OUTPUT);
  Wire.begin();
  Serial.begin(9600);
}
void loop() { /* Page 2 */ }
```  

3. Tools → Board → **Arduino Uno**  
4. Upload → Done  

**ESP32 (Thonny)**  
1. Install Thonny  
2. Tools → Options → Interpreter → **MicroPython (ESP32)**  
3. Copy `dunes_qubit_core.py` from GitHub to ESP32  
4. Run → See LED blink  

---

### First Test: Blink a Qubit  

**Arduino sketch (add to loop):**  
```cpp
digitalWrite(2, HIGH); delay(500);
digitalWrite(2, LOW);  delay(500);
```  

**ESP32 script:**  
```python
import machine, time
led = machine.Pin(2, machine.Pin.OUT)
while True:
    led.value(1); time.sleep(0.5)
    led.value(0); time.sleep(0.5)
```  

Both LEDs blink? **You’re live.**  

---

**Next:** **Page 2 – Code the Qubit Magic**  

*© 2025 WebXOS – Build, flash, glow!* ✨
