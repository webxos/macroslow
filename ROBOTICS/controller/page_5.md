# **ESP32 Full-Screen Robot Rover Commander**  
### *Turn Your ESP32 into a Pro Tank-Style Mobile Controller — No Coding Needed*  
**Page 5 / 10**

---

## **PAGE 5: DOWNLOAD CODE & HTML FILES**

### **OPTION 1: DOWNLOAD ZIP (RECOMMENDED)**
1. Click:  
   🔗 **[Download ROVER_FULL_APP.zip](https://example.com/rover-download)**  
   *(Real link in final build)*  
2. Extract → Folder: `ROVER_FULL_APP/`

> **Must contain:**  
> `ROVER_FULL_APP.ino` + `data/index.html`

---

### **OPTION 2: COPY-PASTE (NO DOWNLOAD)**

#### **File 1: `ROVER_FULL_APP.ino`**
```cpp
// ROVER_FULL_APP.ino
#include <WiFi.h>
#include <AsyncTCP.h>
#include <ESPAsyncWebServer.h>
#include <LittleFS.h>
#include <WebSocketsServer.h>
#include <ElegantOTA.h>
#include <ArduinoJson.h>
#include <DNSServer.h>
#include "esp_camera.h"

#define AP_SSID "ROVER_CTRL"
#define AP_PASS "tankgo"
#define LEFT_M1 12
#define LEFT_M2 13
#define RIGHT_M1 14
#define RIGHT_M2 15
#define PAN_PIN 2
#define TILT_PIN 4

AsyncWebServer server(80);
WebSocketsServer ws(81);
DNSServer dns;

void setup() {
  Serial.begin(115200);
  LittleFS.begin();
  pinMode(LEFT_M1, OUTPUT); pinMode(LEFT_M2, OUTPUT);
  pinMode(RIGHT_M1, OUTPUT); pinMode(RIGHT_M2, OUTPUT);
  pinMode(PAN_PIN, OUTPUT); pinMode(TILT_PIN, OUTPUT);

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
      if(type == "motor"){
        int l = doc["l"], r = doc["r"];
        digitalWrite(LEFT_M1, l>0); digitalWrite(LEFT_M2, l<0);
        digitalWrite(RIGHT_M1, r>0); digitalWrite(RIGHT_M2, r<0);
      }
      else if(type == "servo"){
        analogWrite(PAN_PIN, map(doc["p"], -100, 100, 90, 180));
        analogWrite(TILT_PIN, map(doc["t"], -100, 100, 90, 180));
      }
      else if(type == "ota") ElegantOTA.begin(&server);
      else if(type == "wifi"){
        WiFi.begin(doc["s"], doc["p"]);
        File f = LittleFS.open("/wifi.json","w"); f.print(p); f.close();
      }
    }
  });
  ws.begin();

  ElegantOTA.begin(&server);

  // Camera init
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5; config.pin_d1 = 18; config.pin_d2 = 19; config.pin_d3 = 21;
  config.pin_d4 = 36; config.pin_d5 = 39; config.pin_d6 = 34; config.pin_d7 = 35;
  config.pin_xclk = 0; config.pin_pclk = 22; config.pin_vsync = 25;
  config.pin_href = 23; config.pin_sscb_sda = 26; config.pin_sscb_scl = 27;
  config.pin_pwdn = 32; config.pin_reset = -1; config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size = FRAMESIZE_VGA; config.jpeg_quality = 12; config.fb_count = 1;
  esp_camera_init(&config);

  server.on("/stream", HTTP_GET, [](AsyncWebServerRequest *req){
    AsyncWebServerResponse *resp = req->beginResponse_P(200, "multipart/x-mixed-replace; boundary=frame", nullptr, 0, 
      [](uint8_t *buffer, size_t maxLen, size_t index) -> size_t {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb) return 0;
        size_t len = snprintf((char*)buffer, maxLen,
          "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n", fb->len);
        if (len >= maxLen) { esp_camera_fb_return(fb); return 0; }
        memcpy(buffer + len, fb->buf, fb->len);
        esp_camera_fb_return(fb);
        return len + fb->len;
      });
    resp->addHeader("Access-Control-Allow-Origin", "*");
    req->send(resp);
  });

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
*(Full tank UI + video feed → on Page 6)*

---

### **FOLDER STRUCTURE**
```
ROVER_FULL_APP/
├── ROVER_FULL_APP.ino
└── data/
    └── index.html
```

---

**Next: Page 6 → Full `index.html` Tank UI + Video**  
*xaiartifacts: ROVER_GUIDE.md (Page 5 – code + camera stream)*

---  
**#ESP32RobotCommander** | *Copy. Paste. Roll.*
