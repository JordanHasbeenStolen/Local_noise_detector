#!/usr/bin/env python3
"""
Run: python monitor.py
Open: http://localhost:9000  or  http://<device-ip>:9000
"""

import time, math, threading
import numpy as np
import pyaudio
from flask import Flask, jsonify, request, Response
from flask_cors import CORS

# ─── Settings ────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16_000
CHUNK_SIZE       = 1_024
DEFAULT_THRESHOLD = 500
ALERT_HOLD_SECS  = 3.0
PORT             = 9000

# ─── Shared state (audio stream ↔ HTTP) ──────────────────────────────────────
state = {"alert": False, "rms": 0.0, "db": -90.0, "last_trigger": 0.0}
config = {"threshold": DEFAULT_THRESHOLD}
lock   = threading.Lock()

# ─── Audio thread ────────────────────────────────────────────────────────────
def audio_thread():
    pa = pyaudio.PyAudio()
    info = pa.get_default_input_device_info()
    print(f"[mic]  {info['name']}")
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                     input=True, frames_per_buffer=CHUNK_SIZE)
    print(f"[mic]  Listening… port={PORT}")
    try:
        while True:
            raw     = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            rms     = math.sqrt(np.mean(samples ** 2))
            db      = 20 * math.log10(rms + 1e-9)
            now     = time.time()
            with lock:
                thr = config["threshold"]
                state["rms"] = rms
                state["db"]  = db
                if rms >= thr:
                    state["last_trigger"] = now
                state["alert"] = (now - state["last_trigger"]) < ALERT_HOLD_SECS
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop_stream(); stream.close(); pa.terminate()

# ─── HTML (embedded in Python) ───────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Room Monitor</title>
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --green:#00e676;--red:#ff1744;--bg:#0d0d0d;
  --border:#2a2a2a;--text:#e0e0e0;--muted:#555;
  --font:'SF Mono','Fira Mono','Consolas',monospace;
}
html,body{height:100%;background:var(--bg);color:var(--text);font-family:var(--font);overflow:hidden}
body{display:grid;grid-template-rows:auto 1fr auto;height:100vh}

/* header */
header{
  display:flex;align-items:center;justify-content:space-between;
  padding:.7rem 1.4rem;border-bottom:1px solid var(--border);
  font-size:.7rem;letter-spacing:.15em;text-transform:uppercase;color:var(--muted)
}
#conn-dot{
  display:inline-block;width:7px;height:7px;border-radius:50%;
  background:var(--muted);margin-right:.5rem;transition:background .3s
}
#conn-dot.ok{background:var(--green)}
#conn-dot.err{background:var(--red)}

/* center */
main{
  display:flex;flex-direction:column;align-items:center;
  justify-content:center;gap:1.2rem;overflow:hidden;padding:.5rem
}
#lamp{
  width:min(36vw,36vh);height:min(36vw,36vh);flex-shrink:0;
  border-radius:50%;background:var(--green);cursor:pointer;
  box-shadow:0 0 55px 15px color-mix(in srgb,var(--green) 45%,transparent),
             0 0 110px 35px color-mix(in srgb,var(--green) 18%,transparent);
  transition:background .3s,box-shadow .3s
}
#lamp.alert{
  background:var(--red);
  box-shadow:0 0 55px 15px color-mix(in srgb,var(--red) 45%,transparent),
             0 0 110px 35px color-mix(in srgb,var(--red) 18%,transparent)
}
#status-label{
  font-size:clamp(.9rem,2.5vw,1.3rem);letter-spacing:.3em;
  text-transform:uppercase;color:var(--green);transition:color .3s
}
#status-label.alert{color:var(--red)}

#db-display{
  font-size:clamp(2rem,5vw,3.5rem);font-weight:700;
  color:var(--text);letter-spacing:-.02em;line-height:1
}
#db-display span{font-size:.4em;color:var(--muted);margin-left:.2em}

/* chart */
#chart-wrap{width:min(88vw,680px)}
canvas#chart{width:100%;height:80px;display:block}
#chart-labels{
  display:flex;justify-content:space-between;
  font-size:.6rem;color:var(--muted);margin-top:.25rem
}

/* footer */
footer{
  border-top:1px solid var(--border);padding:.8rem 1.4rem;
  display:flex;align-items:center;gap:2rem;flex-wrap:wrap
}
.ctrl{display:flex;flex-direction:column;gap:.3rem}
.ctrl label{font-size:.62rem;letter-spacing:.13em;text-transform:uppercase;color:var(--muted)}
.ctrl-row{display:flex;align-items:center;gap:.8rem}

input[type=range]{
  -webkit-appearance:none;appearance:none;
  width:200px;height:4px;border-radius:2px;outline:none;cursor:pointer;
  background:linear-gradient(to right,var(--green) 0%,var(--green) var(--pct,50%),#2a2a2a var(--pct,50%))
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:14px;height:14px;
  border-radius:50%;background:var(--text);cursor:pointer
}
#thr-value{font-size:.85rem;min-width:5ch;color:var(--text)}

#btn-fs{
  margin-left:auto;padding:.4rem .9rem;
  background:transparent;border:1px solid var(--border);border-radius:6px;
  color:var(--muted);font-family:var(--font);font-size:.65rem;
  letter-spacing:.1em;text-transform:uppercase;cursor:pointer;
  transition:border-color .2s,color .2s
}
#btn-fs:hover{border-color:#555;color:var(--text)}

#ka{position:fixed;bottom:0;left:0;width:1px;height:1px;opacity:.01;pointer-events:none}
</style>
</head>
<body>

<header>
  <div><span id="conn-dot"></span><span id="conn-text">connecting…</span></div>
  <div>room monitor · 9000</div>
</header>

<main>
  <div id="lamp" title="click = fullscreen"></div>
  <div id="status-label">—</div>
  <div id="db-display">—<span>dB</span></div>
  <div id="chart-wrap">
    <canvas id="chart" width="680" height="80"></canvas>
    <div id="chart-labels"><span>60 sec ago</span><span>now</span></div>
  </div>
</main>

<footer>
  <div class="ctrl">
    <label>Trigger threshold (lower → more sensitive)</label>
    <div class="ctrl-row">
      <input type="range" id="thr-slider" min="40" max="90" step="1" value="65"/>
      <span id="thr-value">65 dB</span>
    </div>
  </div>
  <button id="btn-fs">⛶ Fullscreen</button>
</footer>

<canvas id="ka" width="1" height="1"></canvas>

<script>
/* ── Keep-awake ── */
(function(){
  const c=document.getElementById('ka'),ctx=c.getContext('2d');
  let h=0;
  function tick(){h=(h+1)%360;ctx.fillStyle=`hsl(${h},50%,50%)`;ctx.fillRect(0,0,1,1);requestAnimationFrame(tick)}
  tick();
  try{const s=c.captureStream(1),v=document.createElement('video');v.srcObject=s;v.muted=true;v.loop=true;v.play().catch(()=>{})}catch(e){}
})();

/* ── Chart ──
   Real microphone dB values are usually in the 30–80 range.
   Draw the scale from 30 to 90 dB.
*/
const HISTORY=120, DB_MIN=30, DB_MAX=90;
const dbHistory=new Array(HISTORY).fill(DB_MIN);
const chartCanvas=document.getElementById('chart');
const cctx=chartCanvas.getContext('2d');
let currentThresholdDb=65;

function dbToY(db,H){
  return H - (db-DB_MIN)/(DB_MAX-DB_MIN)*H;
}

function drawChart(alertNow){
  const W=chartCanvas.width, H=chartCanvas.height;
  cctx.clearRect(0,0,W,H);

  // grid
  cctx.lineWidth=1;
  [40,50,60,70,80].forEach(db=>{
    const y=dbToY(db,H);
    cctx.strokeStyle='#1e1e1e';
    cctx.beginPath();cctx.moveTo(0,y);cctx.lineTo(W,y);cctx.stroke();
    cctx.fillStyle='#383838';cctx.font='9px monospace';
    cctx.fillText(db+' dB',4,y-3);
  });

  // sound line
  cctx.beginPath();
  dbHistory.forEach((db,i)=>{
    const x=i/(HISTORY-1)*W;
    const y=dbToY(Math.max(DB_MIN,Math.min(DB_MAX,db)),H);
    i===0?cctx.moveTo(x,y):cctx.lineTo(x,y);
  });
  cctx.strokeStyle=alertNow?'#ff1744':'#00e676';
  cctx.lineWidth=1.5;cctx.stroke();

  // threshold line
  const ty=dbToY(currentThresholdDb,H);
  cctx.setLineDash([5,4]);
  cctx.strokeStyle='#ff6d00';cctx.lineWidth=1.5;
  cctx.beginPath();cctx.moveTo(0,ty);cctx.lineTo(W,ty);cctx.stroke();
  cctx.setLineDash([]);
  cctx.fillStyle='#ff6d00';cctx.font='9px monospace';
  cctx.fillText('threshold '+currentThresholdDb+' dB', 4, ty-4);
}

/* ── Slider ── */
const slider=document.getElementById('thr-slider');
const thrValue=document.getElementById('thr-value');

function rmsFromDb(db){ return Math.pow(10,db/20); }

function updateTrack(){
  const pct=((slider.value-slider.min)/(slider.max-slider.min)*100).toFixed(1)+'%';
  slider.style.setProperty('--pct',pct);
  currentThresholdDb=parseInt(slider.value);
  thrValue.textContent=currentThresholdDb+' dB';
}

let debounce=null;
slider.addEventListener('input',()=>{
  updateTrack();
  clearTimeout(debounce);
  debounce=setTimeout(()=>{
    fetch('/threshold',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({value:rmsFromDb(currentThresholdDb)})}).catch(()=>{});
  },200);
});
updateTrack();
// send initial value
fetch('/threshold',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({value:rmsFromDb(currentThresholdDb)})}).catch(()=>{});

/* ── Polling ── */
const lamp=document.getElementById('lamp');
const label=document.getElementById('status-label');
const dbDisp=document.getElementById('db-display');
const connDot=document.getElementById('conn-dot');
const connTxt=document.getElementById('conn-text');

async function poll(){
  try{
    const res=await fetch('/status',{signal:AbortSignal.timeout(1200)});
    const d=await res.json();
    const isAlert=!!d.alert;

    lamp.classList.toggle('alert',isAlert);
    label.classList.toggle('alert',isAlert);
    label.textContent=isAlert?'NOISE':'QUIET';

    const db=typeof d.db==='number'?d.db:DB_MIN;
    dbDisp.innerHTML=`${db.toFixed(1)}<span>dB</span>`;

    dbHistory.push(db);
    if(dbHistory.length>HISTORY)dbHistory.shift();
    drawChart(isAlert);

    connDot.className='ok';connTxt.textContent='connected';
  }catch(e){
    connDot.className='err';connTxt.textContent='no connection';
  }
}
setInterval(poll,500);poll();

/* ── Fullscreen ── */
[document.getElementById('btn-fs'),lamp].forEach(el=>{
  el.addEventListener('click',()=>{
    if(!document.fullscreenElement)document.documentElement.requestFullscreen().catch(()=>{});
    else document.exitFullscreen().catch(()=>{});
  });
});
</script>
</body>
</html>"""

# ─── Flask ───────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    return Response(HTML, mimetype="text/html")

@app.route("/status")
def status():
    with lock:
        return jsonify({"alert": state["alert"], "rms": round(state["rms"],1),
                        "db": round(state["db"],1)})

@app.route("/threshold", methods=["POST"])
def set_threshold():
    data = request.get_json(silent=True) or {}
    val  = data.get("value")
    if isinstance(val, (int, float)) and 0 < val <= 32767:
        with lock:
            config["threshold"] = float(val)
        return jsonify({"ok": True, "threshold": config["threshold"]})
    return jsonify({"ok": False, "error": "value must be 10–10000"}), 400

# ─── Startup ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=audio_thread, daemon=True)
    t.start()
    print(f"[web]  http://localhost:{PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)
