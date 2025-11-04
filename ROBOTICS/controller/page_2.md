# **ESP32 Full-Screen Robot Rover Commander**  
### *Turn Your ESP32 into a Pro Tank-Style Mobile Controller — No Coding Needed*  
**Page 2 / 10**

---

## **PAGE 2: WHAT YOU NEED (HARDWARE & SOFTWARE)**

### **HARDWARE (BASE: $12 — NO SOLDERING)**
| Item | Details | Why |
|------|--------|-----|
| **ESP32 Dev Board** | Any (NodeMCU, DevKit) | Runs server + controls motors |
| **ESP32-CAM** | AI-Thinker module | Live MJPEG video stream |
| **Android Phone** | Chrome browser | Fullscreen tank HUD |
| **USB Cable** | Micro-USB (x2) | Flash + power |
| **Computer** | Win/Mac/Linux | Arduino IDE |

> **No soldering for video + basic control.**

---

### **ROBOT CHASSIS (OPTIONAL ADD-ONS)**
| Add-On | Use |
|-------|-----|
| **TT Motor + Wheels** | Tank tracks (2x motors) |
| **L298N Driver** | Dual H-bridge (GPIO control) |
| **SG90 Servo** | Turret pan/tilt |
| **IR Sensors** | Line following |
| **HC-SR04** | Obstacle avoid |

---

### **SOFTWARE (ALL FREE)**
| Tool | Link | Purpose |
|------|------|--------|
| **Arduino IDE 2.x** | [arduino.cc/en/software](https://www.arduino.cc/en/software) | Upload code |
| **ESP32 Package** | Boards Manager | ESP32 + CAM support |
| **Libraries** | 6 total (async, camera, etc.) | Web + video + control |

---

### **LIBRARY LIST (INSTALL ORDER)**
1. `ESP Async WebServer` → Fast web  
2. `AsyncTCP` → Dependency  
3. `ESP32 Cam` → MJPEG stream  
4. `LittleFS_esp32` → Store HTML  
5. `ElegantOTA` → Wireless update  
6. `ArduinoJson` → Joystick parse  

---

### **PROJECT FILES**
- `ROVER_FULL_APP.ino` → Main logic  
- `data/index.html` → Tank UI + video  
- Folder: `ROVER_FULL_APP/`

---

**Next: Page 3 → Install Arduino IDE & ESP32-CAM Support**  
*xaiartifacts: ROVER_GUIDE.md (Page 2 – hardware matrix + libs)*

---  
**#ESP32RobotCommander** | *Zero extras. Just roll.*
