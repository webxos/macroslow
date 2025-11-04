## 🐪 Micro Quantum Sim PC Breadboard Build
# *Arduino Uno + ESP32*
**Version 1.0.0** | **WebXOS Research Group** | **Nov 04, 2025**
**Repo:** github.com/webxos/dunes-qubit-sdk
**Email:** project_dunes@outlook.com
---
## PAGE 8: Add Servo Lock & Quantum OTP
### New Parts
| Item | Qty | Notes |
|------|-----|-------|
| SG90 Micro Servo | 1 | Physical lock |
| Jumper wires | 3 | PWM control |

**Cost:** ~$2
---
### Wiring
| Servo | Connect To |
|-------|------------|
| Red | 5 V rail |
| Brown | GND |
| Orange | Arduino Pin 9 |

**Pin 9 now PWM → move TX to Pin 8 if conflict**
---
### Arduino Servo + OTP
```cpp
#include <Servo.h>
Servo lock;
byte otp[4]; int otpIdx = 0;

void setup() {
  lock.attach(9); lock.write(0);  // locked
  randomSeed(millis());
}
void genOTP() {
  if(otpIdx < 4) {
    otp[otpIdx++] = random(2);
    qubit[otpIdx-1] = otp[otpIdx-1];
    digitalWrite(2+otpIdx-1, qubit[otpIdx-1]);
  } else if(otpIdx == 4) {
    lock.write(90); tone(6, 1200, 300); otpIdx++;  // unlock
  }
}
```
**I2C cmd `13` → start OTP sequence.**
---
### Verify & Lock
```cpp
void receiveEvent(int n) {
  /* ... */ 
  else if(cmd == 13) { otpIdx = 0; genOTP(); }
  else if(cmd == 14) { if(matchOTP()) lock.write(0); }  // relock
}
bool matchOTP() {
  for(int i=0;i<4;i++) if(qubit[i] != otp[i]) return false;
  return true;
}
```
---
### ESP32 OTP UI
```python
# HTML:
<button onclick="g('otp')">OTP</button><button onclick="g('chk')">CHECK</button>
# Server:
elif req == '/otp': i2c.writeto(0x08, bytes([13]))
elif req == '/chk': i2c.writeto(0x08, bytes([14]))
```
**Click OTP → 4 LEDs show code → twist pot + gates to match → CHECK unlocks.**
---
### OLED OTP View
```cpp
display.print("OTP: ");
for(int i=0;i<otpIdx;i++) display.print(otp[i]);
if(otpIdx > 4) display.print(" UNLOCKED");
```
**Quantum one-time pad → servo clicks open.**
---
**Next:** **Page 9 – Add Temp Sensor & Phase Drift**  
*© 2025 WebXOS – OTP, unlock, secure!* ✨
