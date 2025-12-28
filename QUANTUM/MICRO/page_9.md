## 🐪 Micro Quantum Sim PC Breadboard Build  
# *Arduino Uno + ESP32*  
**Version 1.0.0** | **WebXOS Research Group** | **Nov 04, 2025**  
---  
## PAGE 9: Add Temp Sensor & Phase Drift  
### New Parts  
| Item | Qty | Notes |  
|------|-----|-------|  
| DS18B20 Temp Sensor | 1 | 1-Wire |  
| 4.7 kΩ resistor | 1 | Pull-up |  

**Cost:** ~$2  
---  
### Wiring  
| DS18B20 | Connect To |  
|---------|------------|  
| VCC | 5 V |  
| GND | GND |  
| DQ | Arduino Pin 8 + 4.7 kΩ → 5 V |  

**Reuse Pin 8 (was TX) → Serial comm via I2C only**  
---  
### Arduino Temp + Drift  
```cpp
#include <OneWire.h>  
#include <DallasTemperature.h>  
OneWire oneWire(8);  
DallasTemperature temp(&oneWire);  
float drift = 0.0;  

void setup() {  
  temp.begin();  
}  
void updateDrift() {  
  temp.requestTemperatures();  
  float t = temp.getTempCByIndex(0);  
  if (t > 30.0) {  
    drift += 0.05; tone(6, 3000, 50);  // overheat beep  
  }  
  phase += drift;  
  if (phase > 2*PI) phase -= 2*PI;  
}  
```  
**Call `updateDrift()` every 2s in `loop()`.**  
---  
### Phase Noise Sim  
```cpp
void measureAll() {  
  float noisyPhase = phase + random(-100,100)/1000.0;  
  float p0 = cos(noisyPhase/2)*cos(noisyPhase/2);  
  for(int i=0;i<4;i++) {  
    qubit[i] = (random(1000)/1000.0 > p0) ? 1 : 0;  
    digitalWrite(2+i, qubit[i]);  
  }  
}  
```  
**Temp ↑ → phase drift → qubit errors → BEEP!**  
---  
### ESP32 Temp Dashboard  
```python
# HTML:  
<div id="temp">TEMP: --°C</div>  
# Server:  
elif req == '/temp':  
    try: t = i2c.readfrom(0x08, 4).decode()  
    except: t = "ERR"  
    resp = t  
# Add to requestEvent():  
Wire.write((int)(temp.getTempCByIndex(0)*100));  
# JS: setInterval(fetch('/temp'), 2000) → update  
```  
**Live thermal quantum noise.**  
---  
### OLED Temp View  
```cpp
display.setCursor(0,48);  
display.print("T:"); display.print(temp.getTempCByIndex(0),1); display.print("C");  
```  
**Heat it → watch phase slip → decoherence!**  
---  
**Next:** **Page 10 – Cloud Sync & Final Demo**  
*© 2025 WebXOS* ✨
