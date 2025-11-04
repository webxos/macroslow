# **ESP32 Full-Screen Drone & Smart Home Controller**  
### *Turn Your ESP32 into a Pro Mobile Game-Style App — No Coding Needed*  
**Page 6 / 10**

---

## **PAGE 6: FULL `index.html` UI CODE (COPY INTO `data/index.html`)**

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>ESP32 Drone Controller</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
    body { background:#111; color:#0f0; font-family:Arial; overflow:hidden; touch-action:none; }
    #app { width:100vw; height:100vh; display:flex; flex-direction:column; }
    .top-bar { flex:0 0 50px; background:#000; display:flex; align-items:center; justify-content:space-between; padding:0 15px; font-size:14px; }
    .status { color:#0f0; }
    .btn { background:#222; border:1px solid #0f0; color:#0f0; padding:6px 12px; border-radius:6px; font-size:12px; }
    .btn:active { background:#0f0; color:#000; }
    .controls { flex:1; display:flex; position:relative; }
    .joystick { position:absolute; width:140px; height:140px; border:2px solid #0f0; border-radius:50%; background:rgba(0,255,0,0.1); }
    .knob { position:absolute; width:60px; height:60px; background:#0f0; border-radius:50%; top:50%; left:50%; transform:translate(-50%,-50%); transition:0.05s; box-shadow:0 0 15px #0f0; }
    #left { left:30px; bottom:30px; }
    #right { right:30px; bottom:30px; }
    .btn-grid { position:absolute; top:60px; left:50%; transform:translateX(-50%); display:grid; grid-template-columns:1fr 1fr; gap:15px; }
    .big-btn { width:80px; height:80px; background:#222; border:2px solid #0f0; border-radius:50%; color:#0f0; font-weight:bold; font-size:14px; display:flex; align-items:center; justify-content:center; box-shadow:0 0 10px rgba(0,255,0,0.3); }
    .big-btn:active { background:#0f0; color:#000; }
    .fullscreen-btn { position:absolute; bottom:15px; right:15px; width:40px; height:40px; background:#222; border:1px solid #0f0; border-radius:8px; color:#0f0; font-size:20px; display:flex; align-items:center; justify-content:center; }
    .fullscreen-btn:active { background:#0f0; color:#000; }
    .wifi-setup { position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); display:none; flex-direction:column; padding:20px; z-index:100; }
    .wifi-setup input { margin:10px 0; padding:12px; background:#222; border:1px solid #0f0; color:#0f0; border-radius:6px; font-size:16px; }
    .wifi-setup .btn { width:100%; margin-top:15px; }
    @media (orientation: portrait) { .controls { flex-direction:column; } .joystick { position:relative; margin:20px auto; } #left, #right { left:50%; transform:translateX(-50%); } .btn-grid { top:auto; bottom:180px; } }
  </style>
</head>
<body>
  <div id="app">
    <div class="top-bar">
      <div class="status" id="status">Disconnected</div>
      <div class="btn" id="wifiBtn">WiFi Setup</div>
    </div>
    <div class="controls">
      <div class="joystick" id="left"><div class="knob" id="lknob"></div></div>
      <div class="joystick" id="right"><div class="knob" id="rknob"></div></div>
      <div class="btn-grid">
        <div class="big-btn" id="arm">ARM</div>
        <div class="big-btn" id="disarm">DISARM</div>
        <div class="big-btn" id="mode1">MODE 1</div>
        <div class="big-btn" id="mode2">MODE 2</div>
      </div>
      <div class="fullscreen-btn" id="fs">⛶</div>
    </div>
  </div>

  <div class="wifi-setup" id="wifiSetup">
    <h2 style="color:#0f0; text-align:center;">Connect to Home WiFi</h2>
    <input type="text" id="ssid" placeholder="Network Name (SSID)" />
    <input type="password" id="pass" placeholder="Password" />
    <div class="btn" id="saveWifi">Save & Connect</div>
    <div class="btn" id="closeWifi" style="margin-top:10px; background:#400;">Cancel</div>
  </div>

  <script>
    const ws = new WebSocket('ws://' + location.hostname + ':81');
    const status = document.getElementById('status');
    const ljoy = document.getElementById('left'), rjoy = document.getElementById('right');
    const lknob = document.getElementById('lknob'), rknob = document.getElementById('rknob');
    const fsBtn = document.getElementById('fs');
    const wifiSetup = document.getElementById('wifiSetup');
    const wifiBtn = document.getElementById('wifiBtn');

    let ltouch = null, rtouch = null;
    const ppm = [1500,1500,1500,1500,1000,1000,1500,1500]; // ch0-7

    ws.onopen = () => status.textContent = 'Connected';
    ws.onclose = () => status.textContent = 'Disconnected';

    function send(type, data) { ws.send(JSON.stringify({t:type, ...data})); }

    function updatePPM() {
      send('ppm', {c:0, v:ppm[0]}); // throttle
      send('ppm', {c:1, v:ppm[1]}); // roll
      send('ppm', {c:2, v:ppm[2]}); // pitch
      send('ppm', {c:3, v:ppm[3]}); // yaw
      send('ppm', {c:4, v:ppm[4]}); // arm
      send('ppm', {c:5, v:ppm[5]}); // mode
    }

    function moveKnob(knob, touch, joy) {
      const rect = joy.getBoundingClientRect();
      const cx = rect.left + rect.width/2, cy = rect.top + rect.height/2;
      let dx = touch.clientX - cx, dy = touch.clientY - cy;
      const dist = Math.min(Math.sqrt(dx*dx + dy*dy), 40);
      const angle = Math.atan2(dy, dx);
      dx = dist * Math.cos(angle); dy = dist * Math.sin(angle);
      knob.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
      return {x: dx/40, y: -dy/40}; // -1 to 1
    }

    ljoy.addEventListener('touchstart', e => { if(!ltouch) ltouch = e.changedTouches[0]; e.preventDefault(); }, {passive:false});
    rjoy.addEventListener('touchstart', e => { if(!rtouch) rtouch = e.changedTouches[0]; e.preventDefault(); }, {passive:false});

    document.addEventListener('touchmove', e => {
      for(let t of e.changedTouches) {
        if(ltouch && t.identifier === ltouch.identifier) {
          const {x,y} = moveKnob(lknob, t, ljoy);
          ppm[0] = Math.round(1000 + 1000 * (1 - y)); // throttle (up = high)
          ppm[3] = Math.round(1000 + 1000 * x);       // yaw
          ltouch = t;
        }
        if(rtouch && t.identifier === rtouch.identifier) {
          const {x,y} = moveKnob(rknob, t, rjoy);
          ppm[1] = Math.round(1000 + 1000 * x);       // roll
          ppm[2] = Math.round(1000 + 1000 * y);       // pitch (forward = high)
          rtouch = t;
        }
      }
      updatePPM();
      e.preventDefault();
    }, {passive:false});

    document.addEventListener('touchend', e => {
      for(let t of e.changedTouches) {
        if(ltouch && t.identifier === ltouch.identifier) { ltouch = null; lknob.style.transform = 'translate(-50%,-50%)'; ppm[0]=1500; ppm[3]=1500; }
        if(rtouch && t.identifier === rtouch.identifier) { rtouch = null; rknob.style.transform = 'translate(-50%,-50%)'; ppm[1]=1500; ppm[2]=1500; }
      }
      updatePPM();
    });

    ['arm','disarm','mode1','mode2'].forEach(id => {
      document.getElementById(id).addEventListener('click', () => {
        if(id === 'arm') ppm[4] = 2000;
        if(id === 'disarm') ppm[4] = 1000;
        if(id === 'mode1') ppm[5] = 1500;
        if(id === 'mode2') ppm[5] = 2000;
        updatePPM();
        setTimeout(() => { if(ppm[4]!==1500) ppm[4]=1500; if(ppm[5]!==1000) ppm[5]=1000; updatePPM(); }, 300);
      });
    });

    fsBtn.onclick = () => document.documentElement.requestFullscreen();
    wifiBtn.onclick = () => wifiSetup.style.display = 'flex';
    document.getElementById('closeWifi').onclick = () => wifiSetup.style.display = 'none';
    document.getElementById('saveWifi').onclick = () => {
      const s = document.getElementById('ssid').value;
      const p = document.getElementById('pass').value;
      if(s) send('wifi', {s, p});
      wifiSetup.style.display = 'none';
    };

    // OTA
    document.getElementById('status').ondblclick = () => send('ota', {});
  </script>
</body>
</html>
```

---

**COPY THIS ENTIRE CODE** → Save as: `data/index.html`  
*(Must be inside `data/` folder)*

---

**Next: Page 7 → Upload Code & Web App to ESP32**  
*xaiartifacts: FULL_GUIDE.md (Page 6 – complete `index.html` with dual joysticks, PPM, fullscreen, WiFi setup)*

---  
**#ESP32GameController** | *Touch. Fly. Control.*
