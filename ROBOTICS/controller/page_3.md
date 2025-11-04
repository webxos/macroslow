# **ESP32 Full-Screen Robot Rover Commander**  
### *Turn Your ESP32 into a Pro Tank-Style Mobile Controller — No Coding Needed*  
**Page 3 / 10**

---

## **PAGE 3: INSTALL ARDUINO IDE & ESP32-CAM SUPPORT**

### **STEP 1: DOWNLOAD & INSTALL ARDUINO IDE (5 MINS)**
1. Open browser → **[https://www.arduino.cc/en/software](https://www.arduino.cc/en/software)**  
2. Click **“Arduino IDE 2.x”** (latest)  
3. Select **Windows / Mac / Linux** → Run installer  
4. Click **“Next → Install”** (defaults)  
5. Launch → Clean editor opens

---

### **STEP 2: ADD ESP32 BOARD PACKAGE**
1. **File → Preferences** (`Ctrl + ,`)  
2. **Additional Boards Manager URLs**: Paste  
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```  
3. Click **OK**

---

### **STEP 3: INSTALL ESP32 + CAM SUPPORT**
1. **Tools → Board → Boards Manager**  
2. Search: `esp32`  
3. Install: **“esp32 by Espressif Systems”** (v2.0.17+)  
   → Includes **ESP32-CAM** drivers  
4. Wait 3–5 mins (~250MB)

---

### **STEP 4: SELECT BOARD**
1. **Tools → Board → ESP32 Arduino**  
2. Choose: **“AI Thinker ESP32-CAM”**  
   *(For video streaming)*

---

### **STEP 5: PARTITION (CRITICAL FOR HTML + VIDEO)**
1. **Tools → Partition Scheme**  
2. Select: **“Huge APP (3MB No OTA)”**  
   *(Fits web app + camera buffer)*

---

### **QUICK TEST: CAMERA BLINK**
1. **File → Examples → ESP32 → Camera → CameraWebServer**  
2. Connect **ESP32-CAM** via FTDI/USB  
3. **Tools → Port →** Select COM  
4. **Upload** → Open Serial Monitor  
   → See **MJPEG stream URL**

---

**Next: Page 4 → Install Required Libraries**  
*xaiartifacts: ROVER_GUIDE.md (Page 3 – CAM board + partition)*

---  
**#ESP32RobotCommander** | *Camera ready in 10 mins.*
