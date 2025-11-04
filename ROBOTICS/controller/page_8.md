# **ESP32 Full-Screen Robot Rover Commander**  
### *Turn Your ESP32 into a Pro Tank-Style Mobile Controller — No Coding Needed*  
**Page 8 / 10**

---

## **PAGE 8: CONNECT PHONE & LAUNCH TANK HUD**

### **STEP 1: POWER ESP32-CAM**
1. **Unplug FTDI/USB**  
2. Connect **5V + GND** from battery or **LDO regulator**  
   → **3.3V to VCC**, **GND to GND**

> **ESP32-CAM now runs standalone.**

---

### **STEP 2: CONNECT TO ROVER WIFI**
1. **Phone Settings → WiFi**  
2. Find:  
   **📶 `ROVER_CTRL`**  
3. Tap → Password:  
   **`tankgo`**  
4. Connected → IP: `192.168.4.x`

> **No internet = normal.** Local command center.

---

### **STEP 3: OPEN TANK COMMANDER**
1. Open **Chrome**  
2. Address bar:  
   ```
   192.168.4.1
   ```
3. Enter → **Cinematic HUD loads**  
   → **Live MJPEG video** appears instantly

---

### **STEP 4: GO FULLSCREEN (BATTLE MODE)**
1. Tap **⛶** (bottom-right)  
2. **“Go Fullscreen”** → **No bars. Pure immersion**

> **Rotate to landscape** → Max video + giant tracks

---

### **ADD TO HOME SCREEN (INSTANT LAUNCH)**
1. Chrome **⋮ → Add to Home Screen**  
2. Name: `Rover Commander`  
3. → **Military-style app icon**

---

### **TEST CONTROLS**
| Action | Result |
|-------|--------|
| **Left Track Up** | Left motor forward |
| **Right Track Down** | Right motor reverse |
| **Turret Knob** | Pan + Tilt (servo) |
| **Video Feed** | Real-time MJPEG stream |

> **Wiring:**  
> Motors → L298N → ESP32 pins 12–15  
> Servos → 3.3V + Pin 2, 4

---

### **STATUS CHECK**
| Indicator | Meaning |
|---------|--------|
| **“Connected”** | WebSocket live |
| **Smooth video** | 10–15 FPS (VGA) |
| **Track glow** | Touch active |

---

**Next: Page 9 → How to Drive & Autonomous Modes**  
*xaiartifacts: ROVER_GUIDE.md (Page 8 – connection + video HUD)*

---  
**#ESP32RobotCommander** | *Lock. Load. Roll out.*
