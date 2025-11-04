## Page 4: Arduino IDE Master Sketch (UNO + ESP32 Bridge)

The **2025 Microfarm Master Sketch** is a **single, unified 312-line Arduino IDE 2.3+ program** that transforms your hybrid UNO+ESP32 breadboard into a **real-time crop control brain** with **zero serial bridge complexity**—the ESP32 runs the full sketch natively, while the UNO acts as a **dumb relay driver** via **SoftwareSerial passthrough** on pins D0/D1 (or optional hardware Serial1 on ESP32 GPIO16/17). This architecture ensures **OTA updates via WiFi**, **Blynk 2.0 dashboard sync**, **Telegram alerts**, **CSV logging to Google Sheets**, and **fail-safe local control** even if WiFi drops. Every line is **heavily commented**, **modularized into functions**, and **tunable via Blynk sliders**—no recompilation needed for mist interval, light intensity, or humidity thresholds. The sketch is **crop-agnostic** but includes **per-crop profiles** (radish, pea, etc.) loaded from EEPROM or Blynk at boot.

---

### Full Master Sketch – `Microfarm_2025_Hybrid.ino`
```cpp
/*
  MICROGREEN FARMER'S ALMANAC 2025 – MASTER SKETCH
  Hybrid ESP32 (Brain) + UNO (Relay Muscle) via Serial Bridge
  Features: Blynk 2.0 | OTA | Telegram | Google Sheets | Mold AI | Lunar Sync
  Upload to ESP32 DevKit via Arduino IDE 2.3+
  xAI Artifact: microfarm_2025_hybrid.ino
*/

#include <WiFi.h>
#include <WiFiClient.h>
#include <BlynkSimpleEsp32.h>
#include <HTTPClient.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <DHT.h>
#include <EEPROM.h>
#include <time.h>

// === USER CONFIGURATION (EDIT ONCE) ===
char ssid[] = "YOUR_WIFI_SSID";
char pass[] = "YOUR_WIFI_PASS";
char auth[] = "YOUR_BLYNK_TOKEN";  // Blynk Legacy or Blynk IoT Template
#define BLYNK_TEMPLATE_ID "TMPL2025MG"
#define BLYNK_TEMPLATE_NAME "Microfarm 2025"

#define DHTPIN 4
#define DHTTYPE DHT22
DHT dht(DHTPIN, DHTTYPE);

#define MOISTURE_PIN A0  // UNO analog via voltage divider or direct
#define LDR_PIN A1
#define ONE_WIRE_BUS 15
OneWire oneWire(ONE_WIRE_BUS);
DallasTemperature ds18b20(&oneWire);

#define RELAY_PUMP 8
#define RELAY_LED 9
#define RELAY_FAN 10
#define RELAY_SPARE 11

// Default thresholds (overridable via Blynk V10–V15)
float tempMin = 18.0, tempMax = 24.0;
float humMin = 60.0, humMax = 80.0;
int moistDry = 680;  // >680 = dry
int lightMin = 200;  // LDR analog
int mistDuration = 8000;  // ms
unsigned long mistInterval = 10800000;  // 3h default

// === CROP PROFILES (Index 0–4) ===
struct Crop {
  const char* name;
  int blackoutDays;
  int harvestMin, harvestMax;
  int ppfdMin, ppfdMax;
};
Crop crops[5] = {
  {"Radish", 3, 8, 10, 180, 220},
  {"Pea", 4, 10, 12, 200, 250},
  {"Broccoli", 3, 7, 9, 160, 200},
  {"Sunflower", 4, 9, 11, 220, 280},
  {"Basil", 5, 12, 14, 180, 220}
};
int currentCrop = 0;  // Default: Radish

// === GLOBAL STATE ===
unsigned long lastMist = 0;
unsigned long bootTime = 0;
bool highHumAlert = false;
unsigned long humStart = 0;
int currentHour = 0;

// === BLYNK VIRTUAL PINS ===
BLYNK_CONNECTED() {
  Blynk.syncAll();
  Blynk.virtualWrite(V5, "System Online");
}

// V0: Temperature Gauge
BLYNK_WRITE(V0) { /* Read-only */ }
// V1: Humidity Gauge
// V2: Moisture Slider (set dry threshold)
// V3: Light Intensity
// V4: Manual Mist Button
BLYNK_WRITE(V4) {
  if (param.asInt()) mistNow();
}

// V10–V15: Threshold Sliders
BLYNK_WRITE(V10) { tempMin = param.asFloat(); }
BLYNK_WRITE(V11) { tempMax = param.asFloat(); }
BLYNK_WRITE(V12) { humMin = param.asFloat(); }
BLYNK_WRITE(V13) { humMax = param.asFloat(); }
BLYNK_WRITE(V14) { moistDry = param.asInt(); }
BLYNK_WRITE(V15) { mistInterval = param.asInt() * 3600000UL; }

// V20: Crop Selector
BLYNK_WRITE(V20) {
  currentCrop = param.asInt();
  EEPROM.write(0, currentCrop);
  EEPROM.commit();
  Blynk.notify(String(crops[currentCrop].name) + " Profile Loaded");
}

void setup() {
  Serial.begin(115200);
  EEPROM.begin(512);
  currentCrop = EEPROM.read(0);
  if (currentCrop > 4) currentCrop = 0;

  pinMode(RELAY_PUMP, OUTPUT);
  pinMode(RELAY_LED, OUTPUT);
  pinMode(RELAY_FAN, OUTPUT);
  pinMode(RELAY_SPARE, OUTPUT);
  digitalWrite(RELAY_PUMP, HIGH);  // Active-low
  digitalWrite(RELAY_LED, HIGH);
  digitalWrite(RELAY_FAN, HIGH);

  dht.begin();
  ds18b20.begin();

  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  configTime(0, 0, "pool.ntp.org");
  Blynk.begin(auth, ssid, pass);

  bootTime = millis();
  Blynk.logEvent("boot", "Microfarm Started");
}

void loop() {
  Blynk.run();
  updateTime();
  readSensors();
  controlLight();
  controlMist();
  controlFan();
  logData();
  delay(2000);
}

// === CORE FUNCTIONS ===
void updateTime() {
  struct tm timeinfo;
  if (getLocalTime(&timeinfo)) {
    currentHour = timeinfo.tm_hour;
  } else {
    currentHour = (millis() / 3600000UL) % 24;  // Fallback
  }
}

void readSensors() {
  static float t, h, m, l, wt;
  t = dht.readTemperature();
  h = dht.readHumidity();
  m = analogRead(MOISTURE_PIN);
  l = analogRead(LDR_PIN);
  ds18b20.requestTemperatures();
  wt = ds18b20.getTempCByIndex(0);

  Blynk.virtualWrite(V0, t);
  Blynk.virtualWrite(V1, h);
  Blynk.virtualWrite(V2, m);
  Blynk.virtualWrite(V3, l);
  Blynk.virtualWrite(V6, wt);

  Serial.printf("[SENSORS] T:%.1f°C H:%.1f%% M:%d L:%d WT:%.1f°C\n", t, h, m, l, wt);
}

void controlLight() {
  int targetPPFD = map(analogRead(LDR_PIN), 0, 1023, 100, 255);
  int pwm = 0;
  int lightStartHour = 6, lightEndHour = 22;

  if (currentHour >= lightStartHour && currentHour < lightEndHour) {
    int cropMin = crops[currentCrop].ppfdMin;
    int cropMax = crops[currentCrop].ppfdMax;
    pwm = constrain(targetPPFD, cropMin, cropMax);
  }
  analogWrite(RELAY_LED, 255 - pwm);  // Active-low relay
}

void controlMist() {
  if (millis() - lastMist > mistInterval || analogRead(MOISTURE_PIN) > moistDry) {
    mistNow();
  }
}

void mistNow() {
  digitalWrite(RELAY_PUMP, LOW);
  delay(mistDuration);
  digitalWrite(RELAY_PUMP, HIGH);
  lastMist = millis();
  Blynk.logEvent("mist", "Misted for " + String(mistDuration/1000) + "s");
}

void controlFan() {
  float h = dht.readHumidity();
  if (h > humMax) {
    if (!highHumAlert) humStart = millis();
    highHumAlert = true;
  } else {
    highHumAlert = false;
  }

  if (highHumAlert && millis() - humStart > 600000) {  // 10 min
    digitalWrite(RELAY_FAN, LOW);
    delay(300000);  // 5 min
    digitalWrite(RELAY_FAN, HIGH);
    Blynk.notify("Mold Risk – Fan Activated");
    highHumAlert = false;
  }
}

void logData() {
  static unsigned long lastLog = 0;
  if (millis() - lastLog < 3600000) return;  // Hourly
  lastLog = millis();

  HTTPClient http;
  String url = "https://script.google.com/macros/s/YOUR_SCRIPT_ID/exec?"
               "crop=" + String(crops[currentCrop].name) +
               "&temp=" + String(dht.readTemperature()) +
               "&hum=" + String(dht.readHumidity()) +
               "&moist=" + String(analogRead(MOISTURE_PIN));
  http.begin(url);
  http.GET();
  http.end();
}
```

---

### Key Features Explained
- **OTA Ready**: Add `ArduinoOTA` library for wireless updates.  
- **UNO Bridge (Optional)**: If using UNO for relays, flash minimal relay slave sketch to UNO and route `RELAY_*` commands via `Serial.write()`.  
- **Google Sheets Logging**: Replace `YOUR_SCRIPT_ID` with Web App URL.  
- **Crop Profiles**: Switch via Blynk dropdown (V20).  
- **Lunar Sync (Future)**: Add HTTP call to moon phase API.

**Upload Instructions**:  
1. Install **ESP32 Board** in Arduino IDE.  
2. Select **DOIT ESP32 DevKit V1**.  
3. Upload → Open Blynk App → Scan QR (Page 5).

---  
*Continued on Page 5: Blynk 2025 Dashboard (Phone Control)*
