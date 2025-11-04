# **ESP32 Full-Screen Drone & Smart Home Controller**  
### *Turn Your ESP32 into a Pro Mobile Game-Style App — No Coding Needed*  
**Page 4 / 10**

---

## **PAGE 4: INSTALL REQUIRED LIBRARIES**

### **OPEN LIBRARY MANAGER**
1. In **Arduino IDE**:  
   **Sketch → Include Library → Manage Libraries…**  
   *(or `Ctrl + Shift + I`)*

---

### **INSTALL 5 LIBRARIES (ONE BY ONE)**

| # | Library | Search Term | Author | Why Needed |
|---|--------|-------------|--------|-----------|
| 1 | **ESP Async WebServer** | `ESP Async WebServer` | *me-no-dev* | Serves web app fast |
| 2 | **AsyncTCP** | `AsyncTCP` | *me-no-dev* | Required for async server |
| 3 | **LittleFS_esp32** | `LittleFS` | *lorol* | Stores `index.html` on ESP32 |
| 4 | **ElegantOTA** | `ElegantOTA` | *Ayush Sharma* | Wireless firmware updates |
| 5 | **ArduinoJson** | `ArduinoJson` | *Benoit Blanchon* | Parses joystick data |

---

### **INSTALL STEPS (REPEAT FOR EACH)**
1. Type **search term** in filter box
2. Click library → **“Install”** (latest version)
3. Wait for **“Installed”** badge
4. **Close** when all 5 are green

> **Pro Tip:** Install in order above — avoids dependency errors.

---

### **VERIFY INSTALLATION**
1. Go to: **Sketch → Include Library**
2. All 5 should appear **without red errors**

---

### **COMMON FIXES**
| Issue | Fix |
|------|-----|
| “Library not found” | Restart Arduino IDE |
| Install stuck | Check internet, try again |
| Version conflict | Uninstall old → reinstall |

---

### **YOU’RE NOW 80% READY**  
All tools + libraries installed.  
Next: **Download the full code & UI**.

---

**Next: Page 5 → Download Code & HTML Files**  
*xaiartifacts: FULL_GUIDE.md (Page 4 complete – library matrix + install flow)*

---  
**#ESP32GameController** | *5 clicks. Done.*
