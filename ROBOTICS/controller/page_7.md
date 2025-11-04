# **ESP32 Full-Screen Robot Rover Commander**  
### *Turn Your ESP32 into a Pro Tank-Style Mobile Controller — No Coding Needed*  
**Page 7 / 10**

---

## **PAGE 7: UPLOAD CODE & WEB APP TO ESP32-CAM**

### **FOLDER CHECK (EXACT)**
```
ROVER_FULL_APP/
├── ROVER_FULL_APP.ino      ← Main code (Page 5)
└── data/
    └── index.html          ← Tank UI + video (Page 6)
```

> **Create `data` folder if missing.**

---

### **STEP 1: OPEN PROJECT**
1. **Arduino IDE → File → Open**  
2. Select: `ROVER_FULL_APP.ino`  
   → Loads `.ino` + `data/` automatically

---

### **STEP 2: BOARD & SETTINGS**
| Setting | Value |
|--------|-------|
| **Board** | `AI Thinker ESP32-CAM` |
| **Upload Speed** | `921600` |
| **Partition Scheme** | `Huge APP (3MB No OTA)` |
| **PSRAM** | `Enabled` *(if board has it)* |
| **Port** | Your COM/USB (via FTDI) |

> **No USB port?** Use **FTDI programmer** (3.3V)

---

### **STEP 3: UPLOAD MAIN CODE**
1. Click **Upload** (⬆️)  
2. **Hold IO0 to GND** → Press **Reset**  
3. Wait:  
   ```
   Compiling... → Writing flash...
   ```
4. **Success?** ESP32-CAM reboots

---

### **STEP 4: UPLOAD WEB FILES (LITTLEFS)**
1. **Tools → ESP32 Sketch Data Upload**  
   *(Install plugin: [github.com/me-no-dev/arduino-esp32fs-plugin](https://github.com/me-no-dev/arduino-esp32fs-plugin))*  
2. Wait:  
   ```
   LittleFS Upload Complete
   ```
3. `index.html` now on flash

---

### **VERIFY (SERIAL MONITOR)**
- **115200 baud** → See:  
  ```
  AP started: ROVER_CTRL
  Camera init done
  ```

---

### **COMMON FIXES**
| Problem | Fix |
|--------|-----|
| “Failed to connect” | Hold **IO0 = GND** during upload |
| No port | Use **3.3V FTDI**, RX→TX, TX→RX |
| Upload timeout | Lower speed → `115200` |
| LittleFS missing | Install plugin + restart IDE |

---

**Next: Page 8 → Connect Phone & Launch Tank HUD**  
*xaiartifacts: ROVER_GUIDE.md (Page 7 – CAM upload + FTDI wiring)*

---  
**#ESP32RobotCommander** | *Flash. Stream. Roll.*
