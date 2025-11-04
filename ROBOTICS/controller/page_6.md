```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no" />
  <title>ESP32 Rover Commander</title>
  <style>
    * { margin:0; padding:0; box-sizing:border-box; -webkit-tap-highlight-color:transparent; }
    body { background:#111; color:#0f0; font-family:Arial; overflow:hidden; touch-action:none; }
    #app { width:100vw; height:100vh; display:flex; flex-direction:column; }
    .top-bar { flex:0 0 50px; background:#000; display:flex; align-items:center; justify-content:space-between; padding:0 15px; font-size:14px; }
    .status { color:#0f0; }
    .btn { background:#222; border:1px solid #0f0; color:#0f0; padding:6px 12px; border-radius:6px; font-size:12px; }
    .btn:active { background:#0f0; color:#000; }
    .video { flex:0 0 40%; background:#000; position:relative; }
    #cam { width:100%; height:100%; object-fit:contain; }
    .controls { flex:1; display:flex; position:relative; }
    .track { position:absolute; width:130px; height:200px; border:2px solid #0f0; border-radius:20px; background:rgba(0,255,0,0.1); }
    .knob { position:absolute; width:60px; height:60px; background:#0f0; border-radius:50%; top:50%; left:50%; transform:translate(-50%,-50%); transition:0.05s; box-shadow:0 0 15px #0f0; }
    #left { left:30px; top:50%; transform:translateY(-50%); }
    #right { right:30px; top:50%; transform:translateY(-50%); }
    .turret { position:absolute; bottom:30px; left:50%; transform:translateX(-50%); width:100px; height:100px; border:2px solid #0f0; border-radius:50%; background:rgba(0,255,0,0.1); }
    .turret-knob { position:absolute; width:40px; height:40px; background:#0f0; border-radius:50%; top:50%; left:50%; transform:translate(-50%,-50%); transition:0.05s; }
    .btn-grid { position:absolute; top:15px; left:50%; transform:translateX(-50%); display:grid; grid-template-columns:1fr 1fr; gap:12px; }
    .big-btn { width:70px; height:70px; background:#222; border:2px solid #0f0; border-radius:50%; color:#0f0; font-weight:bold; font-size:12px; display:flex; align-items:center; justify-content:center; box-shadow:0 0 10px rgba(0,255,0,0.3); }
    .big-btn:active { background:#0f0; color:#000; }
    .fs-btn { position:absolute; bottom:15px; right:15px; width:40px; height:40px; background:#222; border:1px solid #0f0; border-radius:8px; color:#0f0; font-size:20px; display:flex; align-items:center; justify-content:center; }
    .fs-btn:active { background:#0f0; color:#000; }
    .wifi-setup { position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.95); display:none; flex-direction:column; padding:20px; z-index:100; }
    .wifi-setup input { margin:10px 0; padding:12px; background:#222; border:1px solid #0f0; color:#0f0; border-radius:6px; font-size:16px; }
    .wifi-setup .btn { width:100%; margin-top:15px; }
    @media (orientation: portrait) { .video { flex:0 0 30%; } .track { width:100px; height:150px; } .turret { width:80px; height:80px; } }
  </style>
</head>
<body>
  <div id="app">
    <div class="top-bar">
      <div class="status" id="status">Disconnected</div>
      <div class="btn" id="wifiBtn">WiFi Setup</div>
    </div>
    <div class="video"><img id="cam" src="/stream" /></div>
    <div class="controls">
      <div class="track" id="left"><div class="knob" id="lknob"></div></div>
      <div class="track" id="right"><div class="knob" id="rknob"></div></div>
      <div class="turret" id="turret"><div class="turret-knob" id="tknob"></div></div>
      <div class="btn-grid">
        <div class="big-btn" id="auto">AUTO</div>
        <div class="big-btn" id="patrol">PATROL</div>
        <div class="big-btn" id="lights">LIGHTS</div>
        <div class="big-btn" id="scan">SCAN</div>
      </div>
      <div class="fs-btn" id="fs">⛶</div>
    </div>
  </div>

  <div class="wifi-setup" id="wifiSetup">
    <h2 style="color:#0f0; text-align:center;">Connect to Home WiFi</h2>
    <input type="text" id="ssid" placeholder="SSID" />
    <input type="password" id="pass" placeholder="Password" />
    <div class="btn" id="saveWifi">Save & Connect</div>
    <div class="btn" id="closeWifi" style="margin-top:10px; background:#400;">Cancel</div>
  </div>

  <script>
    const ws = new WebSocket('ws://' + location.hostname + ':81');
    const status = document.getElementById('status');
    const ltrack = document.getElementById('left'), rtrack = document.getElementById('right');
    const lknob = document.getElementById('lknob'), rknob = document.getElementById('rknob');
    const turret = document.getElementById('turret'), tknob = document.getElementById('tknob');
    const fsBtn = document.getElementById('fs');
    const wifiSetup = document.getElementById('wifiSetup');
    const wifiBtn = document.getElementById('wifiBtn');

    let ltouch = null, rtouch = null, ttouch = null;
    let motors = {l:0, r:0}, turretPos = {p:0, t:0};

    ws.onopen = () => status.textContent = 'Connected';
    ws.onclose = () => status.textContent = 'Disconnected';

    function send(type, data) { ws.send(JSON.stringify({t:type, ...data})); }
    function updateMotors() { send('motor', motors); }
    function updateTurret() { send('servo', turretPos); }

    function moveKnob(knob, touch, joy, maxY = 75) {
      const rect = joy.getBoundingClientRect();
      const cx = rect.left + rect.width/2, cy = rect.top + rect.height/2;
      let dy = touch.clientY - cy;
      dy = Math.max(-maxY, Math.min(maxY, dy));
      knob.style.transform = `translate(-50%, calc(-50% + ${dy}px))`;
      return -dy / maxY; // -1 to 1
    }

    function moveTurret(touch) {
      const rect = turret.getBoundingClientRect();
      const cx = rect.left + rect.width/2, cy = rect.top + rect.height/2;
      let dx = touch.clientX - cx, dy = touch.clientY - cy;
      const max = 40;
      dx = Math.max(-max, Math.min(max, dx)); dy = Math.max(-max, Math.min(max, dy));
      tknob.style.transform = `translate(calc(-50% + ${dx}px), calc(-50% + ${dy}px))`;
      turretPos.p = Math.round(dx / max * 100);
      turretPos.t = Math.round(-dy / max * 100);
      updateTurret();
    }

    ltrack.addEventListener('touchstart', e => { if(!ltouch) ltouch = e.changedTouches[0]; e.preventDefault(); }, {passive:false});
    rtrack.addEventListener('touchstart', e => { if(!rtouch) rtouch = e.changedTouches[0]; e.preventDefault(); }, {passive:false});
    turret.addEventListener('touchstart', e => { if(!ttouch) ttouch = e.changedTouches[0]; e.preventDefault(); }, {passive:false});

    document.addEventListener('touchmove', e => {
      for(let t of e.changedTouches) {
        if(ltouch && t.identifier === ltouch.identifier) {
          motors.l = Math.round(moveKnob(lknob, t, ltrack) * 255);
          ltouch = t;
        }
        if(rtouch && t.identifier === rtouch.identifier) {
          motors.r = Math.round(moveKnob(rknob, t, rtrack) * 255);
          rtouch = t;
        }
        if(ttouch && t.identifier === ttouch.identifier) {
          moveTurret(t);
          ttouch = t;
        }
      }
      updateMotors();
      e.preventDefault();
    }, {passive:false});

    document.addEventListener('touchend', e => {
      for(let t of e.changedTouches) {
        if(ltouch && t.identifier === ltouch.identifier) { ltouch = null; lknob.style.transform = 'translate(-50%,-50%)'; motors.l = 0; }
        if(rtouch && t.identifier === rtouch.identifier) { rtouch = null; rknob.style.transform = 'translate(-50%,-50%)'; motors.r = 0; }
        if(ttouch && t.identifier === ttouch.identifier) { ttouch = null; tknob.style.transform = 'translate(-50%,-50%)'; turretPos = {p:0,t:0}; updateTurret(); }
      }
      updateMotors();
    });

    ['auto','patrol','lights','scan'].forEach(id => {
      document.getElementById(id).addEventListener('click', () => {
        // Future: send mode commands
        send('mode', {m: id});
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

    // OTA trigger
    status.ondblclick = () => send('ota', {});
  </script>
</body>
</html>
```

---

**COPY ENTIRE CODE** → Save as: `data/index.html`  
*(Inside `data/` folder)*

---

**Next: Page 7 → Upload Code & Web App to ESP32-CAM**  
*xaiartifacts: ROVER_GUIDE.md (Page 6 – full tank UI + MJPEG video + turret control)*

---  
**#ESP32RobotCommander** | *See. Drive. Conquer.*
