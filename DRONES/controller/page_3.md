# **ESP32 Full-Screen Drone & Smart Home Controller**  
### *Turn Your ESP32 into a Pro Mobile Game-Style App — No Coding Needed*  
**Page 3 / 10**

---

## **PAGE 3: INSTALL ARDUINO IDE & ESP32 SUPPORT**

### **STEP 1: DOWNLOAD & INSTALL ARDUINO IDE (5 MINUTES)**
1. Open browser → Go to:  
   🔗 **[https://www.arduino.cc/en/software](https://www.arduino.cc/en/software)**
2. Click **“Download Arduino IDE 2.x”** (latest stable)
3. Choose your OS: **Windows / Mac / Linux**
4. Run installer → Click **“Next” → “Install”** (default settings)
5. Launch **Arduino IDE** when done

> **Success?** You’ll see a clean editor with “Sketch” menu.

---

### **STEP 2: ADD ESP32 BOARD SUPPORT**
1. In Arduino IDE:  
   **File → Preferences** (or `Ctrl + ,`)
2. Find **“Additional Boards Manager URLs”**
3. Paste this **exact URL** in the box:  
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
4. Click **OK**

---

### **STEP 3: INSTALL ESP32 PACKAGE**
1. Go to: **Tools → Board → Boards Manager**
2. In search bar, type: `esp32`
3. Select: **“esp32 by Espressif Systems”**
4. Click **Install** (v2.0.17+ recommended)
5. Wait ~2–5 mins (downloads ~200MB)

> **Done?** ESP32 boards now appear under **Tools → Board**

---

### **STEP 4: SELECT YOUR BOARD**
1. **Tools → Board → ESP32 Arduino**
2. Choose: **“ESP32 Dev Module”**  
   *(Works with 99% of ESP32 boards)*

---

### **STEP 5: SET PARTITION SCHEME (CRITICAL)**
1. **Tools → Partition Scheme**
2. Select: **“Default 4MB with spiffs (1.2MB APP/1.5MB SPIFFS)”**  
   *(Needed to store `index.html`)*

---

### **QUICK TEST: BLINK LED**
1. **File → Examples → 01.Basics → Blink**
2. Connect ESP32 via USB
3. **Tools → Port →** Select your COM/USB port
4. Click **Upload** (⬆️ arrow)
5. Built-in LED blinks? → **You’re ready!**

---

**Next: Page 4 → Install Required Libraries**  
*xaiartifacts: FULL_GUIDE.md (Page 3 complete – full install steps + visuals)*

---  
**#ESP32GameController** | *IDE ready in 10 mins or less.*
