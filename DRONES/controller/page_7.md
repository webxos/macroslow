# **ESP32 Full-Screen Drone & Smart Home Controller**  
### *Turn Your ESP32 into a Pro Mobile Game-Style App — No Coding Needed*  
**Page 7 / 10**

---

## **PAGE 7: UPLOAD CODE & WEB APP TO ESP32**

### **FOLDER CHECK (MUST MATCH)**
```
ESP32_FULL_APP/
├── ESP32_FULL_APP.ino      ← Main code
└── data/
    └── index.html          ← Full UI (from Page 6)
```

> **No `data` folder?** Create it manually in the same directory.

---

### **STEP 1: OPEN PROJECT**
1. Launch **Arduino IDE**
2. **File → Open** → Select `ESP32_FULL_APP.ino`
3. IDE loads both `.ino` and `data/` automatically

---

### **STEP 2: CONFIGURE BOARD SETTINGS**
| Setting | Value |
|-------|-------|
| **Tools → Board** | `ESP32 Dev Module` |
| **Tools → Upload Speed** | `921600` |
| **Tools → Partition Scheme** | `Default 4MB with spiffs` |
| **Tools → Port** | Your ESP32 COM/USB port |

> **Port missing?** Replug USB, check Device Manager.

---

### **STEP 3: UPLOAD MAIN CODE**
1. Click **Upload** (⬆️ arrow)
2. Wait:  
   ```
   Compiling... → Writing flash... → Hard resetting...
   ```
3. **Success?** ESP32 reboots (LED may blink)

---

### **STEP 4: UPLOAD WEB FILES (SPIFFS)**
1. **Tools → ESP32 Sketch Data Upload**  
   *(Install tool if missing: [github.com/me-no-dev/arduino-esp32fs-plugin](https://github.com/me-no-dev/arduino-esp32fs-plugin))*
2. Wait:  
   ```
   SPIFFS Upload Complete
   ```
3. **Done!** `index.html` now stored on ESP32 flash

---

### **VERIFY UPLOAD**
- Serial Monitor (`Ctrl+Shift+M`, 115200 baud):  
  ```
  LittleFS mounted
  AP started: DRONE_CTRL
  ```
- No errors = **Ready to connect**

---

### **COMMON FIXES**
| Problem | Fix |
|-------|-----|
| “Failed to connect” | Hold **BOOT** button during upload |
| SPIFFS tool missing | Install from link above |
| Upload hangs | Try different USB cable/port |
| Wrong partition | Re-select “Default 4MB with spiffs” |

---

**Next: Page 8 → Connect Phone & Go Fullscreen**  
*xaiartifacts: FULL_GUIDE.md (Page 7 – full upload workflow + SPIFFS tool link)*

---  
**#ESP32GameController** | *Flash once. Fly forever.*
