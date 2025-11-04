## 🐪 Micro Quantum Sim PC Breadboard Build
# *Arduino Uno + ESP32*
**Version 1.0.0** | **WebXOS Research Group** | **Nov 04, 2025**
**Repo:** github.com/webxos/dunes-qubit-sdk
**Email:** project_dunes@outlook.com
---
## PAGE 3: WiFi Dashboard & Qiskit Link
### Enhanced ESP32 Web UI (`dunes_qubit_core.py` update)
```python
import machine, network, time
from machine import Pin, SoftI2C

i2c = SoftI2C(scl=Pin(22), sda=Pin(21))
led = Pin(2, Pin.OUT)
ssid = "DUNES-Q"; pw = "qubit123"

html = """<!DOCTYPE html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">
<style>body{font:16px Arial;text-align:center;background:#111;color:#0f0}
button{width:80px;height:60px;margin:10px;font:20px bold;background:#222;color:#0f0;border:2px solid #0f0;border-radius:10px}
</style></head><body><h1>🐪 Dunes Qubit Sim</h1>
<div id="state">---- </div>
<button onclick="g('h0')">H0</button><button onclick="g('h1')">H1</button>
<button onclick="g('h2')">H2</button><button onclick="g('h3')">H3</button><br>
<button onclick="g('c01')">C01</button><button onclick="g('meas')">MEAS</button>
<script>
function g(cmd){fetch('/'+cmd).then(r=>r.text()).then(d=>getState())}
function getState(){fetch('/state').then(r=>r.text()).then(t=>document.getElementById('state').innerText=t)}
setInterval(getState,500); getState();
</script></body></html>"""

def wifi_ap():
    ap = network.WLAN(network.AP_IF)
    ap.active(True); ap.config(essid=ssid, password=pw)
    print("AP:", ssid)

def web_server():
    import socket
    s = socket.socket(); s.bind(('',80)); s.listen(1)
    state = "----"
    while True:
        conn, addr = s.accept()
        req = conn.recv(1024).decode().split()[1]
        resp = "OK"
        if req == '/state':
            try: state = i2c.readfrom(0x08, 1).decode()
            except: state = "ERR"
            resp = state
        elif req.startswith('/h'): i2c.writeto(0x08, bytes([int(req[2])]))
        elif req == '/c01': i2c.writeto(0x08, bytes([0]))
        elif req == '/meas': i2c.writeto(0x08, bytes([10]))
        elif req == '/': resp = html
        conn.send(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n"+resp.encode())
        conn.close()

wifi_ap(); web_server()
```
**Run → `192.168.4.1` → Live dashboard + controls.**
---
### Arduino I2C Slave Update (add to sketch)
```cpp
#define I2C_ADDR 0x08
void setup() { /* ... */ Wire.begin(I2C_ADDR); Wire.onReceive(receiveEvent); Wire.onRequest(requestEvent); }
void receiveEvent(int n) {
  while(Wire.available()) { byte cmd = Wire.read(); if(cmd<4) hadamard(cmd); else if(cmd==10) measureAll(); }
}
void requestEvent() { Wire.write(qubit[0]+'0'); Wire.write(qubit[1]+'0'); Wire.write(qubit[2]+'0'); Wire.write(qubit[3]+'0'); }
```
**Upload → Dashboard shows real-time `|0101>` etc.**
---
### Qiskit Bridge (PC → ESP32)
```python
# dunes_qiskit.py
from qiskit import QuantumCircuit
import urequests as requests

def run_circuit(qc):
    for inst in qc.data:
        op = inst.operation.name
        qb = inst.qubits[0].index
        if op == 'h': requests.get("http://192.168.4.1/h"+str(qb))
        elif op == 'cx': requests.get("http://192.168.4.1/c01")  # fixed CNOT
    requests.get("http://192.168.4.1/meas")
    state = requests.get("http://192.168.4.1/state").text
    print("Result:", state)

qc = QuantumCircuit(2); qc.h(0); qc.cx(0,1); run_circuit(qc)
```
**Run on PC → Qiskit controls breadboard.**
---
**Next:** **Page 4 – Add OLED Display & Grover Search**  
*© 2025 WebXOS – Dash, sim, Qiskit!* ✨
