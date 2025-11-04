# **ESP32 Full-Screen Drone & Smart Home Controller**  
### *Turn Your ESP32 into a Pro Mobile Game-Style App — No Coding Needed*  
**Page 5 / 10**

---

## **PAGE 5: DOWNLOAD CODE & HTML FILES**

### **OPTION 1: DOWNLOAD ZIP (FASTEST)**
1. Click:  
   🔗 **[Download ESP32_FULL_APP.zip](https://example.com/download)**  
   *(Real link in final version)*  
2. Extract to folder: `ESP32_FULL_APP/`

> **Folder must contain:**  
> `ESP32_FULL_APP.ino` + `data/index.html`

---

### **OPTION 2: COPY-PASTE (NO DOWNLOAD)**

#### **File 1: `ESP32_FULL_APP.ino`**
```cpp
// ESP32_FULL_APP.ino
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <LittleFS.h>
#include <WebSocketsServer.h>
#include <ElegantOTA.h>
#include <ArduinoJson.h>
#include <DNSServer.h>

#define AP_SSID "DRONE_CTRL"
#define AP_PASS "flysafe"
#define PPM_PIN 13
#define SERVO_PIN 12
#define LED_PIN 2

AsyncWebServer server(80);
WebSocketsServer ws(81);
DNSServer dns;
hw_timer_t *ppmTimer = NULL;
uint16_t ppm[8] = {1500,1500,1500,1500,1000,1000,1500,1500};
uint8_t ppmIdx = 0;

void setup() {
  Serial.begin(115200);
  LittleFS.begin();
  pinMode(LED_PIN, OUTPUT);
  pinMode(PPM_PIN, OUTPUT);

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAPConfig(IPAddress(192,168,4,1), IPAddress(192,168,4,1), IPAddress(255,255,255,0));
  WiFi.softAP(AP_SSID, AP_PASS);

  dns.start(53, "*", IPAddress(192,168,4,1));

  server.serveStatic("/", LittleFS, "/").setDefaultFile("index.html");

  ws.onEvent([](uint8_t n, WStype_t t, uint8_t* p, size_t l){
    if(t == WStype_TEXT){
      DynamicJsonDocument doc(256);
      deserializeJson(doc, p);
      String type = doc["t"];
      if(type == "ppm") ppm[doc["c"].as<int>()] = doc["v"];
      else if(type == "led") digitalWrite(LED_PIN, doc["v"]);
      else if(type == "ota") ElegantOTA.begin(&server);
      else if(type == "wifi"){
        WiFi.begin(doc["s"], doc["p"]);
        File f = LittleFS.open("/wifi.json","w"); f.print(p); f.close();
      }
    }
  });
  ws.begin();

  ElegantOTA.begin(&server);

  ppmTimer = timerBegin(0, 80, true);
  timerAttachInterrupt(ppmTimer, []() IRAM_ATTR {
    static uint32_t last = 0;
    uint32_t now = micros();
    if(ppmIdx < 8){
      digitalWrite(PPM_PIN, ppmIdx%2==0 ? HIGH : LOW);
      timerAlarmWrite(ppmTimer, ppm[ppmIdx++], true);
    } else {
      digitalWrite(PPM_PIN, LOW);
      timerAlarmWrite(ppmTimer, 22500 - (now-last), true);
      ppmIdx = 0;
    }
    last = now;
  }, true);
  timerAlarmWrite(ppmTimer, 300, true);
  timerAlarmEnable(ppmTimer);

  server.begin();
}

void loop() {
  ws.loop();
  dns.processNextRequest();
  ElegantOTA.loop();
}
```

---

#### **File 2: `data/index.html`**  
*(Copy full HTML from Page 6 — it’s the **latest fullscreen UI**)*

---

### **FOLDER STRUCTURE (MUST MATCH)**
```
ESP32_FULL_APP/
├── ESP32_FULL_APP.ino
└── data/
    └── index.html
```

> **Create `data` folder manually if needed.**

---

**Next: Page 6 → Full `index.html` UI Code**  
*xaiartifacts: FULL_GUIDE.md (Page 5 – code + structure)*

---  
**#ESP32GameController** | *Copy. Paste. Done.*
