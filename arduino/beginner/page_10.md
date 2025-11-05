# Project 10 of 10: ESP32 WiFi Mesh Node  
**Mastery Goal**: PainlessMesh Network & Message Routing  

---

## Overview  
The ESP32 WiFi Mesh project teaches **self-healing wireless mesh** using **PainlessMesh**. Key concepts:  
- Nodes auto-connect in **ad-hoc 2.4GHz WiFi**.  
- Send/receive **JSON messages** (e.g., drone commands).  
- **Gateway node** bridges to phone via BLE/serial.  
- Fallback-ready for hybrid mesh.  

Critical for **multi-drone swarm**, **redundant control**, and **follow-me coordination**.

---

## Components Needed (2+ Nodes)  
| Item | Quantity | Notes |
|------|----------|-------|
| ESP32 Dev Boards | 2–3 | WROOM-32 |
| LED + 220Ω | 1 per node | Status |
| Breadboard & Jumpers | As needed | |
| Phone/PC | 1 | Serial monitor |

---

## Circuit Wiring (Per Node)  

```
+5V ───────●────────────────┐
           │                │
GND ───────┴────●───────────┘
                │
           ┌────┴────┐
           │  LED    │
GPIO2 ────┤─│220│───┤──► Anode
           │ └───┘   │
           └────┬────┘
                │
               GND
```

**Wiring (all nodes)**:  
- LED: GPIO2 → 220Ω → anode → GND  
- Power: USB-C  

> **Node 1 = Gateway** (connect to PC for logs)  
> **Nodes 2+ = Drones**

---

## Arduino Code (Full Sketch) – Upload to **ALL ESP32s**

```cpp
/*
  Project 10: ESP32 PainlessMesh Node
  Mastery: WiFi Mesh & JSON Messaging
  Hardware: ESP32 + LED (GPIO2)
*/

#include <painlessMesh.h>
#include <ArduinoJson.h>

#define MESH_PREFIX     "DroneMesh"
#define MESH_PASSWORD   "secure123"
#define MESH_PORT       5555

painlessMesh mesh;
const int ledPin = 2;
uint32_t myNodeId;

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(115200);
  
  mesh.setDebugMsgTypes(ERROR | STARTUP | CONNECTION);
  mesh.init(MESH_PREFIX, MESH_PASSWORD, MESH_PORT);
  mesh.onReceive(&receivedCallback);
  mesh.onNewConnection(&newConnectionCallback);
  
  myNodeId = mesh.getNodeId();
  Serial.printf("\nMesh Node %u Started\n", myNodeId);
  digitalWrite(ledPin, HIGH);  // ON = connected
}

void loop() {
  mesh.update();
  
  // Send test message every 10s
  static uint32_t lastMsg = 0;
  if (millis() - lastMsg > 10000) {
    DynamicJsonDocument doc(128);
    doc["from"] = myNodeId;
    doc["cmd"] = "ping";
    doc["val"] = random(0, 100);
    
    String msg;
    serializeJson(doc, msg);
    mesh.sendBroadcast(msg);
    
    Serial.printf("Sent: %s\n", msg.c_str());
    lastMsg = millis();
  }
  
  // Blink if isolated
  if (mesh.getNodeList().size() == 0) {
    digitalWrite(ledPin, (millis() % 1000 < 100) ? HIGH : LOW);
  } else {
    digitalWrite(ledPin, HIGH);
  }
}

void receivedCallback(uint32_t from, String &msg) {
  Serial.printf("Received from %u: %s\n", from, msg.c_str());
  
  DynamicJsonDocument doc(128);
  deserializeJson(doc, msg);
  
  if (doc["cmd"] == "ping") {
    digitalWrite(ledPin, !digitalRead(ledPin));  // Flash on ping
    delay(100);
    digitalWrite(ledPin, HIGH);
  }
}

void newConnectionCallback(uint32_t nodeId) {
  Serial.printf("New node %u joined\n", nodeId);
}
```

---

## Upload & Test (2+ ESP32s)

1. **Install Library**:  
   - **painlessMesh** by **BlackEdder** (Library Manager)  
2. Upload same code to **all ESP32s**.  
3. Power via USB.  
4. Open **Serial Monitor** (115200 baud) on **each**.  

**Expected Behavior**:  
- Nodes auto-connect → **LED solid ON**.  
- Every 10s:  
  ```
  Sent: {"from":12345678,"cmd":"ping","val":42}
  Received from 87654321: {"from":87654321,"cmd":"ping","val":77}
  ```
- LED **flashes** on incoming ping.  
- Move nodes → mesh **self-heals**.

---

## Troubleshooting  

| Issue | Cause | Fix |
|------|-------|-----|
| Not connecting | WiFi channel clash | Reboot all |
| LED blinking | No peers | Bring closer (<10m) |
| JSON error | Buffer small | Increase doc size |
| Only one node | Same USB port | Use separate ports |

---

## Mastery Checkpoint ✅  
You have mastered WiFi mesh if:  
- [ ] **2+ nodes** connect and exchange pings.  
- [ ] **LED solid** when linked, **flash** on message.  
- [ ] You can change `MESH_PREFIX` and reconnect.  

**Next Challenge**: Add **BLE bridge** on gateway to forward mesh → phone.

---

**All 10 Projects Complete!**  
You now have full sensor, control, and mesh mastery for the **bodycam follow-drone**.  
*All artifacts updated for Page 10.*
