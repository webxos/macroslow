# **ESP32 Full-Screen Robot Rover Commander**  
### *Turn Your ESP32 into a Pro Tank-Style Mobile Controller — No Coding Needed*  
**Page 9 / 10**

---

## **PAGE 9: HOW TO DRIVE & AUTONOMOUS MODES**

### **TANK-STYLE DRIVING (DUAL TRACKS)**

| Control | Action | Motor Output |
|--------|--------|--------------|
| **Left Track ↑** | Forward | `LEFT_M1 = HIGH`, `LEFT_M2 = LOW` |
| **Left Track ↓** | Reverse | `LEFT_M1 = LOW`, `LEFT_M2 = HIGH` |
| **Right Track ↑** | Forward | `RIGHT_M1 = HIGH`, `RIGHT_M2 = LOW` |
| **Right Track ↓** | Reverse | `RIGHT_M1 = LOW`, `RIGHT_M2 = HIGH` |
| **Both Up** | **Straight** | Full speed |
| **One Up, One Down** | **Pivot Turn** | Zero-radius spin |
| **Both Center** | **Stop** | Motors off |

> **Speed:** Knob position = 0–255 PWM (smooth control)

---

### **TURRET CONTROL (PAN + TILT)**

| Knob Move | Servo | Angle |
|----------|-------|-------|
| **Left/Right** | Pan | 90° → 180° |
| **Up/Down** | Tilt | 90° → 180° |
| **Center** | Neutral | 135° both |

> **Servos on Pin 2 (Pan), Pin 4 (Tilt)** — 3.3V logic

---

### **AUTONOMOUS MODES (ONE-TAP)**

| Button | Mode | Future Action |
|-------|------|---------------|
| **AUTO** | Line Follow | IR sensors → follow black line |
| **PATROL** | Obstacle Avoid | HC-SR04 → random path |
| **LIGHTS** | Toggle LEDs | Headlights on/off |
| **SCAN** | 360° Sweep | Turret auto-rotate |

> **Code-ready:** `send('mode', {m: 'auto'})` → expand in `.ino`

---

### **DRIVING TIPS**
- **Landscape = Best View** → Video top, tracks bottom  
- **Pivot Turn** → One track forward, one reverse  
- **Turret Lock** → Hold knob for precision aim  
- **Video Lag?** → Lower resolution in code (`FRAMESIZE_QVGA`)

---

### **WIRING SUMMARY**
```
ESP32-CAM → L298N
Pin 12 → IN1 (Left)
Pin 13 → IN2
Pin 14 → IN3 (Right)
Pin 15 → IN4

Servos:
Pin 2 → Pan Signal
Pin 4 → Tilt Signal
```

---

**Next: Page 10 → Upgrades, Troubleshooting & Final Mission**  
*xaiartifacts: ROVER_GUIDE.md (Page 9 – tank physics + modes)*

---  
**#ESP32RobotCommander** | *Drive. Aim. Conquer.*
