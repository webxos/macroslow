## 🐪 Micro Quantum Sim PC Breadboard Build
# *Arduino Uno + ESP32*
**Version 1.0.0** | **WebXOS Research Group** | **Nov 04, 2025**
---
## PAGE 5: Teleport & Error Beep
### Teleport Protocol (3-qubit)
```cpp
void bellPair(int a, int b) {
  hadamard(a); cnot(a, b);
}
void teleport(int src, int bellA, int bellB) {
  cnot(src, bellA); hadamard(src);
  byte m1 = qubit[src], m2 = qubit[bellA];
  if (m2) cnot(bellB, bellA);  // X correction
  if (m1) cnot(bellB, src);   // Z correction
  qubit[src] = 0;  // collapse sender
}
```
**I2C cmd `12` → teleport Q0 → Q2 via Bell Q1-Q2.**
---
### Arduino Update
```cpp
void receiveEvent(int n) {
  while(Wire.available()) {
    byte cmd = Wire.read();
    if(cmd<4) hadamard(cmd);
    else if(cmd==10) measureAll();
    else if(cmd==11) applyGrover();
    else if(cmd==12) { bellPair(1,2); teleport(0,1,2); }
  }
}
```
**Upload → Q0 state moves to Q2.**
---
### ESP32 Web Add
```python
# HTML:
<button onclick="g('tele')">TELE</button>
# Server:
elif req == '/tele': i2c.writeto(0x08, bytes([12]))
```
**Click → Q0 vanishes, Q2 matches.**
---
### Error Beep Logic
```cpp
void measureAll() {
  bool error = false;
  for(int i=0;i<4;i++) {
    float p0 = cos(phase/2)*cos(phase/2);
    if (random(1000)/1000.0 > p0) qubit[i] = 1;
    else { qubit[i] = 0; if(i==0) error = true; }
    digitalWrite(2+i, qubit[i]);
  }
  if (error) tone(BUZZER, 2000, 50);  // high beep on collapse error
}
```
**Phase twist → wrong collapse → BEEP!**
---
### OLED Teleport View
- **Q:** `0000` → `1000` → `0010`  
- **P:** `90°`  
- **TELE** → state jumps

**Quantum teleport on breadboard.**
---
**Next:** **Page 6 – Add Sensor Input & BB84 Crypto**  
*© 2025 WebXOS* ✨
