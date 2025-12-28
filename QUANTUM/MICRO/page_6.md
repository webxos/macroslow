## 🐪 Micro Quantum Sim PC Breadboard Build
# *Arduino Uno + ESP32*
**Version 1.0.0** | **WebXOS Research Group** | **Nov 04, 2025**
---
## PAGE 6: Add Sensor Input & BB84 Crypto
### New Parts
| Item | Qty | Notes |
|------|-----|-------|
| Photoresistor (LDR) | 1 | Light → bit |
| 10 kΩ resistor | 1 | Pull-down |

**Cost:** ~$1
---
### Wiring
| LDR | Connect To |
|-----|------------|
| Leg 1 | 5 V |
| Leg 2 | A3 + 10 kΩ → GND |

**Bright light → A3 > 600 → bit 1**
---
### Arduino BB84 Core
```cpp
byte key[8]; int keyIdx = 0;
void genBB84Bit() {
  if(keyIdx >= 8) return;
  int val = analogRead(A3);
  byte bit = (val > 600);
  byte basis = random(2);  // 0=Z, 1=X
  qubit[0] = bit;
  if(basis) { hadamard(0); phase = PI/2; }  // X basis
  key[keyIdx++] = bit;
  digitalWrite(2, qubit[0]);
}
```
**Call `genBB84Bit()` every 1s in `loop()` via timer.**
---
### Key Transmit (I2C)
```cpp
void requestEvent() {
  if(keyIdx==8) for(int i=0;i<8;i++) Wire.write(key[i]+'0');
}
```
**ESP32 pulls full key on demand.**
---
### ESP32 BB84 Dashboard
```python
# HTML add:
<button onclick="g('bb84')">BB84</button><div id="k">KEY: ----</div>
# Server:
elif req == '/bb84':
    try: k = i2c.readfrom(0x08,8).decode()
    except: k = "WAIT"
    resp = k
# JS: after fetch → document.getElementById('k').innerText="KEY: "+t
```
**Shine light → 8-bit key → secure share.**
---
### OLED Key View
```cpp
display.print("BB84: ");
for(int i=0;i<keyIdx;i++) display.print(key[i]);
```
**Live key generation on screen.**
---
**Next:** **Page 7 – RFID Voter ID & Quantum Ballot**  
*© 2025 WebXOS* ✨
