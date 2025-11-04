# **ESP32 Full-Screen Drone & Smart Home Controller**  
### *Turn Your ESP32 into a Pro Mobile Game-Style App — No Coding Needed*  
**Page 8 / 10**

---

## **PAGE 8: CONNECT PHONE & GO FULLSCREEN**

### **STEP 1: POWER ESP32**
1. Unplug USB from computer  
2. Plug into **phone charger** or **power bank**  
   *(ESP32 now runs standalone)*

---

### **STEP 2: CONNECT PHONE TO ESP32 WIFI**
1. Open **Phone Settings → WiFi**  
2. Find network:  
   **📶 `DRONE_CTRL`**  
3. Tap → Enter password:  
   **`flysafe`**  
4. Connected? → IP: `192.168.4.x`

> **No internet? Normal.** This is a local hotspot.

---

### **STEP 3: OPEN WEB APP**
1. Open **Chrome** (or any browser)  
2. Type in address bar:  
   ```
   192.168.4.1
   ```
3. Press **Enter**  
   → **Pro controller loads instantly**

---

### **STEP 4: GO FULLSCREEN (IMMERSIVE MODE)**
1. Tap **⛶** (bottom-right corner)  
2. Tap **“Go Fullscreen”**  
   → **No browser bars. Pure game-style UI**

> **Pro Tip:** Rotate phone → **Landscape mode** for giant joysticks

---

### **ADD TO HOME SCREEN (OPTIONAL)**
1. Chrome menu (⋮) → **“Add to Home Screen”**  
2. Name: `Drone Controller`  
3. → **App icon on desktop!** (like real app)

---

### **TEST JOYSTICKS**
| Action | Result |
|------|--------|
| Move **Left Stick Up** | Throttle ↑ |
| Move **Right Stick Forward** | Pitch forward |
| Tap **ARM** | PPM Channel 5 → 2000ms |

> Connect **Pin 13 → Flight Controller PPM Input**  
> Use 3.3V → GND shared

---

### **QUICK STATUS CHECK**
| Indicator | Meaning |
|---------|--------|
| **“Connected”** (top-left) | WebSocket active |
| **Glowing joysticks** | Touch registered |
| **WiFi Setup button** | Save home WiFi |

---

**Next: Page 9 → How to Use (Drone + Smart Home Modes)**  
*xaiartifacts: FULL_GUIDE.md (Page 8 – connection + fullscreen flow)*

---  
**#ESP32GameController** | *Tap. Fly. Dominate.*
