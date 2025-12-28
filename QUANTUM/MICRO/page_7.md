## 🐪 Micro Quantum Sim PC Breadboard Build
# *Arduino Uno + ESP32*
**Version 1.0.0** | **WebXOS Research Group** | **Nov 04, 2025**
---
## PAGE 7: RFID Voter ID & Quantum Ballot
### New Parts
| Item | Qty | Notes |
|------|-----|-------|
| MFRC522 RFID Module | 1 | 13.56 MHz |
| RFID Card/Tag | 1+ | Voter ID |

**Cost:** ~$3
---
### Wiring (SPI)
| MFRC522 | Arduino |
|---------|---------|
| 3.3 V | 3.3 V rail |
| GND | GND |
| RST | Pin 7 |
| SDA (SS) | Pin 10 |
| MOSI | Pin 11 |
| MISO | Pin 12 |
| SCK | Pin 13 |

**Buzzer on Pin 12 → move to Pin 6**
---
### Arduino RFID + Vote
```cpp
#include <SPI.h>
#include <MFRC522.h>
#define SS_PIN 10
#define RST_PIN 7
MFRC522 rfid(SS_PIN, RST_PIN);

byte vote = 0;  // 0 or 1
String voterID = "";

void setup() {
  SPI.begin(); rfid.PCD_Init();
  pinMode(6, OUTPUT);  // buzzer moved
}
void checkVote() {
  if (!rfid.PICC_IsNewCardPresent() || !rfid.PICC_ReadCardSerial()) return;
  voterID = "";
  for(byte i=0;i<rfid.uid.size;i++) voterID += String(rfid.uid.uidByte[i], HEX);
  vote = qubit[0];  // Q0 = vote bit
  tone(6, vote?1500:800, 200);
  rfid.PICC_HaltA();
}
```
**Call in `loop()` every 500ms.**
---
### I2C Vote Broadcast
```cpp
void requestEvent() {
  if(vote) { Wire.write("VOTE:"); Wire.write(voterID.c_str()); Wire.write(vote+'0'); }
}
```
**ESP32 reads → logs vote.**
---
### ESP32 Vote Dashboard
```python
# HTML:
<button onclick="g('vote')">CAST</button><div id="log">VOTES:</div>
# Server:
elif req == '/vote':
    try: v = i2c.readfrom(0x08, 32).decode()
    except: v = ""
    resp = v
# JS: append to log
```
**Tap card → vote cast → logged.**
---
### OLED Vote View
```cpp
display.print("ID:"); display.print(voterID.substring(0,8));
display.setCursor(0,40); display.print("VOTE:"); display.print(vote);
```
**Quantum-secure anonymous vote.**
---
**Next:** **Page 8 – Add Servo Lock & Quantum OTP**  
*© 2025 WebXOS* ✨
