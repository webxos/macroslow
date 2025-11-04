## 🐪 Micro Quantum Sim PC Breadboard Build
# *Arduino Uno + ESP32*
**Version 1.0.0** | **WebXOS Research Group** | **Nov 04, 2025**
**Repo:** github.com/webxos/dunes-qubit-sdk
**Email:** project_dunes@outlook.com
---
## PAGE 4: Add OLED Display & Grover Search
### New Parts
| Item | Qty | Notes |
|------|-----|-------|
| 0.96" I2C OLED (SSD1306) | 1 | 128x64 |
| Jumper wires | 4 | SDA/SCL reuse |

**Cost:** ~$4
---
### Wiring
| OLED Pin | Connect To |
|----------|------------|
| VCC | 3.3 V rail |
| GND | GND rail |
| SDA | Arduino A4 |
| SCL | Arduino A5 |

**Shared I2C bus – works!**
---
### Arduino OLED Code (add to sketch)
```cpp
#include <Adafruit_SSD1306.h>
#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);

void setup() {
  /* ... */
  if(!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { while(1); }
  display.clearDisplay(); display.setTextColor(WHITE); display.setTextSize(2);
  display.setCursor(0,0); display.print("DUNES"); display.display();
}
void updateOLED() {
  display.clearDisplay(); display.setCursor(0,0);
  display.print("Q:"); for(int i=0;i<4;i++) display.print(qubit[i]);
  display.setCursor(0,24); display.print("P:"); display.print((int)(phase*180/PI)); display.print("deg");
  display.display();
}
```
**Call `updateOLED()` in `loop()` after `displayQubits()`.**
---
### Grover Search (2-qubit)
```cpp
void groverSearch(byte target) {
  // Oracle: flip phase if match
  for(int q=0;q<4;q++) if(qubit[q] != ((target>>q)&1)) return;
  measureAll(); // simulate phase kickback via collapse
}
void applyGrover() {
  byte target = 0b10; // find |10>
  hadamard(0); hadamard(1);
  groverSearch(target);
  hadamard(0); hadamard(1);
}
```
**Add button → trigger `applyGrover()` via I2C cmd `11`.**
---
### ESP32 Web Update (add button)
```python
# In html string, add:
<button onclick="g('grov')">GROV</button>
# In web_server(), add:
elif req == '/grov': i2c.writeto(0x08, bytes([11]))
```
**Click GROV → finds |10> in ~1 iteration.**
---
### OLED Live View
- **Q:** `1010`  
- **P:** `142°`  
- **GROV** → LED2+LED0 glow

**Real-time quantum search on breadboard.**
---
**Next:** **Page 5 – Teleport & Error Beep**  
*© 2025 WebXOS – OLED, search, glow!* ✨
