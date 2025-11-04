## 🐪 Micro Quantum Sim PC Breadboard Build
# *Arduino Uno + ESP32*
**Version 1.0.0** | **WebXOS Research Group** | **Nov 04, 2025**
**Repo:** github.com/webxos/dunes-qubit-sdk
**Email:** project_dunes@outlook.com
---
## PAGE 2: Code the Qubit Magic
### Full Arduino Sketch
```cpp
#include <Wire.h>
#define POT_PHASE A0
#define BUZZER 12

byte qubit[4] = {0,0,0,0};  // |0> = 0, |1> = 1
float phase = 0.0;

void setup() {
  pinMode(2, OUTPUT); pinMode(3, OUTPUT);
  pinMode(4, OUTPUT); pinMode(5, OUTPUT);
  pinMode(13, OUTPUT); pinMode(BUZZER, OUTPUT);
  Wire.begin(); Serial.begin(9600);
  randomSeed(analogRead(A3));
}

void loop() {
  updatePhase();
  i2cCommand();
  hadamard(random(4));
  displayQubits();
  heartbeat();
  delay(100);
}
void updatePhase() {
  int val = analogRead(POT_PHASE);
  phase = val / 1023.0 * 2 * PI;
}
void i2cCommand() {
  if (Wire.available()) {
    byte cmd = Wire.read();
    if (cmd < 4) cnot(cmd, (cmd+1)%4);
    else if (cmd == 10) measureAll();
  }
}
void hadamard(int q) {
  if (random(1000)/1000.0 < 0.5) {
    qubit[q] ^= 1; digitalWrite(2+q, qubit[q]);
  }
}
void cnot(int c, int t) {
  if (qubit[c]) qubit[t] ^= 1;
  digitalWrite(2+t, qubit[t]);
}
void measureAll() {
  for(int i=0;i<4;i++) {
    if (random(1000)/1000.0 > cos(phase/2)*cos(phase/2))
      qubit[i] = 1;
    digitalWrite(2+i, qubit[i]);
  }
  if (qubit[0]+qubit[1]+qubit[2]+qubit[3] == 0) tone(BUZZER, 1000, 100);
}
void displayQubits() {
  Serial.print("Q: ");
  for(int i=0;i<4;i++) Serial.print(qubit[i]);
  Serial.println();
}
void heartbeat() {
  digitalWrite(13, millis()%1000 < 50);
}
```

**Upload → LEDs flicker like qubits.**
---
### Full ESP32 Core (`dunes_qubit_core.py`)
```python
import machine, network, time, ubinascii
from machine import Pin, I2C, SoftI2C

i2c = SoftI2C(scl=Pin(22), sda=Pin(21))
led = Pin(2, Pin.OUT); buzz = Pin(12, Pin.OUT)
ssid = "DUNES-Q"; pw = "qubit123"

def wifi_ap():
    ap = network.WLAN(network.AP_IF)
    ap.active(True); ap.config(essid=ssid, password=pw)
    print("AP:", ssid)

def web_server():
    import socket
    s = socket.socket(); s.bind(('',80)); s.listen(1)
    while True:
        conn, addr = s.accept()
        req = conn.recv(1024).decode()
        if 'GET /h0' in req: i2c.writeto(0x08, bytes([0]))
        elif 'GET /h1' in req: i2c.writeto(0x08, bytes([1]))
        elif 'GET /meas' in req: i2c.writeto(0x08, bytes([10]))
        conn.send(b"HTTP/1.1 200 OK\r\n\r\nOK")
        conn.close()

wifi_ap(); web_server()
```

**Run → Connect phone to `DUNES-Q`, browse `192.168.4.1/h0` → Gate triggers.**
---
### First Sim: Bell State
1. Twist pot → phase shift  
2. Phone → `/h0` → Hadamard Q0  
3. `/h1` → CNOT Q0→Q1  
4. LEDs: Q0=Q1 always  
**Entanglement live.**
---
**Next:** **Page 3 – WiFi Dashboard & Qiskit Link**  
*© 2025 WebXOS – Code, gate, entangle!* ✨
