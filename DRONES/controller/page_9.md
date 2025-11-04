# **ESP32 Full-Screen Drone & Smart Home Controller**  
### *Turn Your ESP32 into a Pro Mobile Game-Style App — No Coding Needed*  
**Page 9 / 10**

---

## **PAGE 9: HOW TO USE (DRONE + SMART HOME MODES)**

### **DRONE CONTROL MODE (PPM OUTPUT)**

| Control | Action | PPM Channel | Value |
|--------|--------|-------------|-------|
| **Left Joystick ↑↓** | **Throttle** | Ch0 | 1000–2000 |
| **Left Joystick ←→** | **Yaw (Rotate)** | Ch3 | 1000–2000 |
| **Right Joystick ←→** | **Roll (Tilt L/R)** | Ch1 | 1000–2000 |
| **Right Joystick ↑↓** | **Pitch (Fwd/Back)** | Ch2 | 1000–2000 |
| **ARM Button** | Arm drone | Ch4 | 2000 (pulse) |
| **DISARM Button** | Disarm | Ch4 | 1000 (pulse) |
| **MODE 1 / MODE 2** | Flight mode | Ch5 | 1500 / 2000 |

> **Wiring:**  
> `ESP32 Pin 13 (PPM) → Flight Controller PPM Input`  
> Share **GND** and **3.3V** (or 5V if FC supports)

---

### **SMART HOME CONTROL (GPIO)**

| Feature | Pin | How to Use |
|--------|-----|------------|
| **LED On/Off** | Pin 2 | Tap **ARM** twice fast → toggles |
| **Servo (Garage, Lock)** | Pin 12 | Add servo → modify code (see Page 10) |
| **Sensor Read** | Any GPIO | Add DHT22, display in app |

> **Double-tap ARM** → Toggles LED (demo)  
> **WiFi Setup** → Save home network → ESP32 joins automatically

---

### **JOYSTICK MAPPING (VISUAL)**
```
LEFT JOYSTICK           RIGHT JOYSTICK
   ↑ Throttle             ↑ Pitch (forward)
← Yaw →                ← Roll → 
   ↓ Idle                ↓ Pitch (back)
```

---

### **BUTTON BEHAVIOR**
| Button | Tap | Hold | Double-Tap |
|-------|-----|------|------------|
| **ARM** | Arm drone | — | Toggle LED |
| **DISARM** | Disarm | — | — |
| **MODE 1** | Stabilize | — | — |
| **MODE 2** | Acro | — | — |

---

### **PRO TIPS**
- **Landscape mode** = Best joystick size  
- **Add to Home Screen** = Instant launch  
- **OTA Update**: Double-tap **“Connected”** → Upload new firmware  
- **Reboot ESP32**: Unplug/replug if frozen

---

**Next: Page 10 → Upgrades, Troubleshooting & What’s Next**  
*xaiartifacts: FULL_GUIDE.md (Page 9 – full control mapping + GPIO demo)*

---  
**#ESP32GameController** | *One tap. Total control.*
