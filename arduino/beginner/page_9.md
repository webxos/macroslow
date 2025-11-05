# Project 9 of 10: ESP32 BLE Beacon  
**Mastery Goal**: BLE Advertising & Phone Detection  

---

## Overview  
The ESP32 BLE Beacon project teaches **Bluetooth Low Energy** broadcasting. Key concepts:  
- Use **ESP32 BLE library** to advertise custom data.  
- Broadcast **UUID + major/minor** (iBeacon format).  
- Scan with **nRF Connect app** (Android/iOS).  
- Toggle LED on phone detection (simulated).  

Essential for **drone-user pairing**, **follow-me**, and **proximity triggers**.

---

## Components Needed  
| Item | Quantity | Notes |
|------|----------|-------|
| ESP32 Dev Board | 1 | WROOM-32 |
| LED + 220Ω Resistor | 1 | Built-in or external |
| Breadboard & Jumpers | As needed | |
| Android/iOS Phone | 1 | With nRF Connect app |

---

## Circuit Wiring (Breadboard Layout)  

```
Breadboard View:
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

**Step-by-Step Wiring**:  
1. **LED**:  
   - Anode → 220Ω → **GPIO2** (built-in LED on many ESP32 boards)  
   - Cathode → **GND**  
2. **Power**: USB-C (ESP32).

> **Note**: GPIO2 is **built-in LED** on most ESP32 dev boards (active HIGH).

---

## Arduino Code (Full Sketch)  

```cpp
/*
  Project 9: ESP32 BLE iBeacon
  Mastery: BLE Advertising & Detection
  Hardware: ESP32, LED on GPIO2
*/

#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLEAdvertising.h>

#define BEACON_UUID "e2c56db5-dffb-48d2-b060-d0f5a71096e0"  // Random
#define MAJOR 1
#define MINOR 1
#define TX_POWER -59  // dBm

BLEAdvertising *pAdvertising;
const int ledPin = 2;
bool advertising = false;

void setup() {
  pinMode(ledPin, OUTPUT);
  Serial.begin(115200);
  Serial.println("ESP32 BLE Beacon Starting...");

  BLEDevice::init("DroneBeacon-01");
  pAdvertising = BLEDevice::getAdvertising();
  
  setBeacon();
  pAdvertising->start();
  advertising = true;
  digitalWrite(ledPin, HIGH);  // LED ON when advertising
  
  Serial.println("iBeacon Advertising...");
  Serial.println("UUID\tMajor\tMinor\tRSSI(dBm)");
  Serial.printf("%s\t%d\t%d\t%d\n", BEACON_UUID, MAJOR, MINOR, TX_POWER);
}

void setBeacon() {
  BLEBeacon beacon;
  beacon.setProximityUUID(BLEUUID(BEACON_UUID));
  beacon.setMajor(MAJOR);
  beacon.setMinor(MINOR);
  beacon.setSignalPower(TX_POWER);
  
  BLEAdvertisementData advData;
  advData.addData(beacon.getData());
  pAdvertising->setAdvertisementData(advData);
}

void loop() {
  // Blink LED every 2s to show alive
  static unsigned long lastBlink = 0;
  if (millis() - lastBlink > 2000) {
    digitalWrite(ledPin, !digitalRead(ledPin));
    lastBlink = millis();
  }
  
  delay(100);
}
```

---

## Upload & Test  

1. **Setup Arduino IDE**:  
   - Add ESP32 board: `https://dl.espressif.com/dl/package_esp32_index.json`  
   - Board: **ESP32 Dev Module**  
   - Port: Select COM  
2. **Install Libraries**:  
   - **BLE** (built-in with ESP32 core)  
3. Upload code.  
4. Open **Serial Monitor** (115200 baud).  
5. **Open nRF Connect app** → **Scan** → Find:  
   - Name: `DroneBeacon-01`  
   - UUID: `e2c56db5-dffb-48d2-b060-d0f5a71096e0`  
   - RSSI: ~-50 to -90 dBm  

**Expected Result**:  
- LED **blinks every 2s**.  
- Phone detects beacon with **UUID, Major=1, Minor=1**.  

---

## Troubleshooting  

| Issue | Cause | Fix |
|------|-------|-----|
| Not visible | BLE not started | Check Serial |
| LED not blinking | Wrong GPIO | Use GPIO2 |
| Upload fail | Wrong board | Select ESP32 Dev |
| Weak signal | TX power | Adjust `-59` |

---

## Mastery Checkpoint ✅  
You have mastered BLE beacon if:  
- [ ] Phone detects **custom UUID**.  
- [ ] LED **blinks** while advertising.  
- [ ] You can change **MAJOR/MINOR** and rescan.  

**Next Challenge**: Add **button press** to toggle advertising ON/OFF.  

---

**Proceed to Project 10 → ESP32 WiFi Mesh Node** when ready.  
*All artifacts updated for Page 9.*
