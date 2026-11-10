r"""Localhost tool: describe an experiment in words, see the plan, and have the simulated robot perform it.

    .venv-isaac\Scripts\python.exe biotech-lab-robotics\fable_lab\lab_tool\app.py        (then open http://127.0.0.1:8765)

- Planning is rule-based (protocol.py), no language model and no network access.
- "Run" starts run_protocol.py (Isaac Sim, headless) as a separate process: one run at a time, after checking that
  the GPU is free (the project's gpu_guard.sh). The page shows the steps as they finish, a live camera picture, the
  tracked liquid volumes and, at the end, the video.
- Everything is simulated; steps are executed by the scripted experts (motion planning + IK).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol  # noqa: E402

ROOT = HERE.parents[2]                                   # the BiotechRobot folder
ISAAC_PY = ROOT / ".venv-isaac" / "Scripts" / "python.exe"
GUARD = ROOT / "biotech-lab-robotics" / "vision_pipeline" / "isaac_validation" / "gpu_guard.sh"
RUNS = ROOT / "data" / "lab_tool" / "runs"
HOST, PORT = "127.0.0.1", 8765
app = FastAPI(title="Lab bench experiment runner")
job = {"proc": None, "dir": None}


def bash():
    for p in (r"C:\Program Files\Git\bin\bash.exe", r"C:\Program Files\Git\usr\bin\bash.exe"):
        if Path(p).exists():
            return p
    return shutil.which("bash")


def gpu_busy():
    """(busy, reason). The guard exits 0 when someone else is using the GPU."""
    b = bash()
    if not b or not GUARD.exists():
        return True, "could not run the GPU check (bash or gpu_guard.sh not found)"
    try:
        # the guard counts any process of this repo as GPU work: tell it that this web app is not one
        r = subprocess.run([b, str(GUARD).replace("\\", "/")], capture_output=True, text=True, timeout=90,
                           env={**os.environ, "GPU_GUARD_SELF_PID": str(os.getpid())})
    except Exception as e:
        return True, f"GPU check failed: {e!r}"
    lines = [x.strip() for x in r.stdout.strip().splitlines() if x.strip()]
    if r.returncode == 0 and lines and "process" in lines[0]:
        # the guard lists every python process of this repo; the tool's own web processes (this page's server, a
        # second copy of it, share_proxy.py) do not use the GPU
        others = [x for x in lines[1:] if "lab_tool" not in x.replace("\\", "/")]
        if not others:
            return False, "free (only this tool's own web processes are running)"
        return True, others[-1]
    return r.returncode == 0, (lines or [""])[-1]


def running():
    return job["proc"] is not None and job["proc"].poll() is None


def run_dir(name=None):
    if name:
        d = (RUNS / name).resolve()
        return d if d.parent == RUNS.resolve() and d.exists() else None
    return job["dir"]


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.get("/api/bench")
def bench():
    return {"bench": protocol.DEFAULT_BENCH, "examples": protocol.EXAMPLES,
            "limits": {"tip_max_ul": protocol.TIP_MAX_UL, "tube_min_remaining_ul": protocol.MIN_REMAIN_UL,
                       "tube_max_ul": protocol.TUBE_MAX_UL, "well_max_ul": protocol.WELL_MAX_UL, "max_wait_s": protocol.MAX_WAIT_S}}


@app.post("/api/plan")
def plan(body: dict):
    return protocol.plan(str(body.get("text", ""))[:4000])


@app.post("/api/run")
def run(body: dict):
    if running():
        return JSONResponse({"error": "A run is already in progress."}, status_code=409)
    p = protocol.plan(str(body.get("text", ""))[:4000])
    if p["error"] or not p["steps"]:
        return JSONResponse({"error": p["error"] or "Nothing to run.", "plan": p}, status_code=400)
    if not body.get("force"):
        busy, why = gpu_busy()
        if busy:
            return JSONResponse({"error": "The GPU is in use: " + why, "gpu_busy": True}, status_code=409)
    if not ISAAC_PY.exists():
        return JSONResponse({"error": f"Isaac Sim environment not found: {ISAAC_PY}"}, status_code=500)
    d = RUNS / time.strftime("%Y%m%d_%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(json.dumps(p, indent=1))
    log = open(d / "run.log", "w")
    job["proc"] = subprocess.Popen([str(ISAAC_PY), str(HERE / "run_protocol.py"), "--plan", str(d / "plan.json"), "--out", str(d)],
                                   cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
    job["dir"] = d
    return {"run": d.name}


@app.post("/api/stop")
def stop():
    if running():
        subprocess.run(["taskkill", "/PID", str(job["proc"].pid), "/T", "/F"], capture_output=True)
        return {"stopped": True}
    return {"stopped": False}


@app.get("/api/status")
def status(run: str | None = None):
    d = run_dir(run)
    if d is None:
        return {"state": "idle", "running": False}
    f = d / "status.json"
    st = json.loads(f.read_text()) if f.exists() else {"state": "starting", "steps": []}
    alive = running() and d == job["dir"]
    st.update(run=d.name, running=alive, has_video=(d / "protocol.mp4").exists())
    for s in st.get("steps", []):
        s["clip"] = bool(s.get("dir")) and (d / s["dir"] / "clip.mp4").exists()
    if not alive and st.get("state") not in ("done", "failed"):
        st["state"] = "stopped"
        log = d / "run.log"
        st["log_tail"] = log.read_text(errors="replace")[-1500:] if log.exists() else ""
    return st


@app.get("/api/frame")
def frame(run: str | None = None):
    d = run_dir(run)
    f = d / "status.json" if d else None
    if not f or not f.exists():
        return Response(status_code=404)
    st = json.loads(f.read_text())
    k = st.get("current")
    if k is None or not st["steps"][k].get("dir"):
        return Response(status_code=404)
    imgs = sorted((d / st["steps"][k]["dir"] / "images").glob("platform_cam1_*.jpg"))
    if len(imgs) < 2:
        return Response(status_code=404)
    return Response(imgs[-2].read_bytes(), media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@app.get("/api/clip")
def clip(step: int, run: str | None = None):
    d = run_dir(run)
    f = d / "status.json" if d else None
    if not f or not f.exists():
        return Response(status_code=404)
    steps = json.loads(f.read_text())["steps"]
    if not 0 <= step < len(steps) or not steps[step].get("dir") or not (d / steps[step]["dir"] / "clip.mp4").exists():
        return Response(status_code=404)
    return FileResponse(d / steps[step]["dir"] / "clip.mp4", media_type="video/mp4")


@app.get("/api/video")
def video(run: str | None = None):
    d = run_dir(run)
    if d is None or not (d / "protocol.mp4").exists():
        return Response(status_code=404)
    return FileResponse(d / "protocol.mp4", media_type="video/mp4")


@app.get("/api/runs")
def runs():
    out = []
    for d in sorted(RUNS.glob("*"), reverse=True)[:30]:
        f = d / "status.json"
        if f.exists():
            st = json.loads(f.read_text())
            out.append({"run": d.name, "result": st.get("result", st.get("state")), "text": st.get("text", "")[:160],
                        "has_video": (d / "protocol.mp4").exists()})
    return out


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Lab bench experiment runner</title>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#1b1f24;--mute:#667085;--line:#d9dee5;--acc:#1f6feb;--ok:#1a7f37;--bad:#cf222e;--warn:#9a6700}
@media (prefers-color-scheme:dark){:root{--bg:#0f1318;--card:#171c23;--ink:#e6edf3;--mute:#9aa7b4;--line:#2b333d;--acc:#58a6ff;--ok:#3fb950;--bad:#f85149;--warn:#d29922}}
*{box-sizing:border-box}main>div{min-width:0}select{max-width:100%}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,Segoe UI,sans-serif}
header{padding:14px 20px;border-bottom:1px solid var(--line)}h1{font-size:19px;margin:0}header p{margin:4px 0 0;color:var(--mute);font-size:13px}
main{display:grid;grid-template-columns:minmax(320px,1fr) minmax(320px,1.15fr);gap:16px;padding:16px 20px;max-width:1400px;margin:auto}
@media (max-width:900px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin-bottom:16px}
h2{font-size:14px;margin:0 0 8px;color:var(--mute);text-transform:uppercase;letter-spacing:.04em}
textarea{width:100%;min-height:110px;border:1px solid var(--line);border-radius:8px;padding:10px;font:inherit;background:var(--bg);color:var(--ink);resize:vertical}
button,select{font:inherit;border-radius:8px;border:1px solid var(--line);padding:8px 14px;background:var(--card);color:var(--ink);cursor:pointer}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}button:disabled{opacity:.5;cursor:default}
.row{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin-top:10px}
table{border-collapse:collapse;width:100%;font-size:13.5px}td,th{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left}
.note{color:var(--warn);font-size:13px;margin:4px 0}.err{color:var(--bad);font-weight:600;margin:6px 0}
ol{margin:6px 0 0 20px;padding:0}li{margin:2px 0}.cmd{font-weight:600;margin-top:10px}
.st{display:inline-block;min-width:64px;font-size:12px;border-radius:999px;padding:1px 8px;border:1px solid var(--line);color:var(--mute);text-align:center}
.st.ok{color:var(--ok);border-color:var(--ok)}.st.failed{color:var(--bad);border-color:var(--bad)}.st.running{color:var(--acc);border-color:var(--acc)}
img#live,video{width:100%;border-radius:8px;background:#000;display:block}
.small{color:var(--mute);font-size:12.5px}.big{font-size:17px;font-weight:600}
</style></head><body>
<header><h1>Lab bench experiment runner</h1>
<p>Describe an experiment; the simulated robot performs it. Simulation only. Steps are executed by scripted experts (motion planning + IK). Liquids are tracked volumes, the centrifuge does not spin, waits are shortened.</p></header>
<main><div>
<div class="card"><h2>Experiment</h2>
<textarea id="text" spellcheck="false"></textarea>
<div class="row"><select id="examples"><option value="">Examples...</option></select>
<button id="planBtn">Show plan</button><button id="runBtn" class="primary">Run in simulation</button><button id="stopBtn" disabled>Stop</button></div>
<div id="msg"></div></div>
<div class="card"><h2>Plan</h2><div id="plan" class="small">Type a description and press "Show plan".</div></div>
<div class="card"><h2>What is on the bench</h2><div id="bench"></div></div>
</div><div>
<div class="card"><h2>Run</h2><div id="state" class="big">Idle</div><div id="runinfo" class="small"></div>
<div id="clipcap" class="small" style="margin-top:10px"></div>
<video id="clip" muted playsinline style="display:none;margin-top:6px"></video>
<div id="livecap" class="small" style="display:none;margin-top:10px">Right now in the simulator (one picture per second; the simulator runs slower than real time):</div>
<img id="live" alt="live camera picture" style="display:none;margin-top:6px;max-width:320px">
<div id="videocap" class="small" style="display:none;margin-top:10px">Whole run:</div>
<video id="video" controls style="display:none;margin-top:10px"></video></div>
<div class="card"><h2>Liquids (uL, tracked)</h2><div id="liq" class="small">No run yet.</div></div>
<div class="card"><h2>Earlier runs</h2><div id="runs" class="small"></div></div>
</div></main>
<script>
const $=id=>document.getElementById(id);let bench=null,poll=null,shown=null,clipRun=null,nextClip=0,clipBusy=false;
function playClips(s){if(!s.steps||!s.run)return;if(clipRun!==s.run){clipRun=s.run;nextClip=0;clipBusy=false}
 const v=$('clip');if(clipBusy)return;const st=s.steps[nextClip];
 if(st&&st.clip){clipBusy=true;v.src='/api/clip?run='+s.run+'&step='+nextClip;v.style.display='block';
  $('clipcap').textContent='Step '+(nextClip+1)+' of '+s.steps.length+', real speed: '+(st.label||st.prim)+(st.state==='failed'?' (this step failed)':'');
  nextClip++;v.onended=()=>{clipBusy=false;tick()};v.onerror=()=>{clipBusy=false};v.play().catch(()=>{clipBusy=false;v.controls=true})}
 else if(s.running&&nextClip>0){$('clipcap').textContent='Step '+nextClip+' shown. The robot is working on step '+((s.current??nextClip)+1)+'; its clip plays when the step is finished.'}
 else if(s.running){$('clipcap').textContent='Each step is shown as a smooth clip as soon as it is finished.'}}
const esc=s=>String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
async function j(url,body){const r=await fetch(url,body?{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}:{});return [r.status,await r.json()]}
function renderPlan(p,steps){let h='';(p.notes||[]).forEach(n=>h+=`<div class="note">${esc(n)}</div>`);if(p.error)h+=`<div class="err">${esc(p.error)}</div>`;
 const ss=steps||p.steps||[];(p.descriptions||p.commands||[]).forEach((c,i)=>{h+=`<div class="cmd">${i+1}. ${esc(typeof c==='string'?c:JSON.stringify(c))}</div><ol>`;
 ss.filter(s=>s.command===i).forEach(s=>{const lab=s.label||[s.prim.replaceAll('_',' '),s.slot||'',s.tip||'',s.well||'',s.ul?(s.ul+' uL'):'',s.seconds?(s.seconds+' s'):''].join(' ');
 h+=`<li><span class="st ${s.state||''}">${s.state||'planned'}</span> ${esc(lab)}</li>`});h+='</ol>'});$('plan').innerHTML=h||'Nothing understood.'}
async function showPlan(){const [,p]=await j('/api/plan',{text:$('text').value});renderPlan(p);return p}
function liquids(l,planned){if(!l){$('liq').textContent='No run yet.';return}let h='<table><tr><th>Where</th><th>Now</th><th>Planned at the end</th></tr>';
 const names=bench?bench.bench.tubes:{};const keys=new Set([...Object.keys(l.tubes||{}),...Object.keys((planned||{}).tubes||{})]);
 [...keys].sort().forEach(k=>h+=`<tr><td>${esc((names[k]||{}).name||'tube')} (${k})</td><td>${(l.tubes||{})[k]??0}</td><td>${((planned||{}).tubes||{})[k]??''}</td></tr>`);
 const w=new Set([...Object.keys(l.wells||{}),...Object.keys((planned||{}).wells||{})]);[...w].sort().forEach(k=>h+=`<tr><td>tray well ${k}</td><td>${(l.wells||{})[k]??0}</td><td>${((planned||{}).wells||{})[k]??''}</td></tr>`);
 h+=`<tr><td>in the pipette tip</td><td>${l.in_tip??0}</td><td>0</td></tr></table>`;$('liq').innerHTML=h}
async function tick(){const q=shown?('?run='+shown):'';const [,s]=await j('/api/status'+q);const run=s.running;
 $('runBtn').disabled=run;$('stopBtn').disabled=!run;if(s.state==='idle'){return}
 $('state').textContent=(s.result&&!run)?s.result:(s.state||'');$('state').style.color=s.result==='SUCCESS'?'var(--ok)':(s.state==='failed'||s.state==='stopped'||(s.result&&s.result!=='SUCCESS'))?'var(--bad)':'';
 const done=(s.steps||[]).filter(x=>x.state==='ok').length;$('runinfo').textContent=`run ${s.run||''}: ${done} of ${(s.steps||[]).length} steps done`+(s.error?(' — '+s.error):'')+(s.log_tail?(' — log: '+s.log_tail.slice(-300)):'');
 if(s.steps&&s.steps.length)renderPlan({descriptions:s.commands,notes:[],error:null},s.steps);liquids(s.liquids,s.planned_final);playClips(s);
 $('livecap').style.display=run?'block':'none';$('videocap').style.display=(!run&&s.has_video)?'block':'none';
 if(run&&s.current!=null){$('live').style.display='block';$('video').style.display='none';$('live').src='/api/frame'+(q?q+'&':'?')+'t='+Date.now()}
 if(!run&&s.has_video){$('live').style.display='none';const v=$('video');if(v.dataset.run!==s.run){v.src='/api/video?run='+s.run;v.dataset.run=s.run}v.style.display='block'}
 if(!run&&poll&&!shown){clearInterval(poll);poll=null;loadRuns()}}
async function run(force){$('msg').innerHTML='';shown=null;const [code,r]=await j('/api/run',{text:$('text').value,force:!!force});
 if(code!==200){let h=`<div class="err">${esc(r.error||'could not start')}</div>`;if(r.gpu_busy)h+='<button id="forceBtn">Run anyway</button>';$('msg').innerHTML=h;if(r.plan)renderPlan(r.plan);
  if(r.gpu_busy)$('forceBtn').onclick=()=>run(true);return}
 $('state').textContent='starting the simulator (about a minute)';$('video').style.display='none';if(!poll)poll=setInterval(tick,1000)}
async function loadRuns(){const [,rs]=await j('/api/runs');$('runs').innerHTML=rs.length?rs.map(r=>`<div style="margin:4px 0"><a href="#" data-run="${r.run}">${r.run}</a> — <b>${esc(r.result||'')}</b><br>${esc(r.text)}</div>`).join(''):'None yet.';
 document.querySelectorAll('#runs a').forEach(a=>a.onclick=e=>{e.preventDefault();shown=a.dataset.run;tick()})}
(async()=>{const [,b]=await j('/api/bench');bench=b;$('text').value=b.examples[0];b.examples.forEach((e,i)=>{const o=document.createElement('option');o.value=i;o.textContent=e.slice(0,70)+'...';$('examples').appendChild(o)});
 $('examples').onchange=e=>{if(e.target.value!==''){$('text').value=b.examples[e.target.value];showPlan()}};
 let h='<table><tr><th>Rack slot</th><th>Contents</th><th>uL</th></tr>';Object.entries(b.bench.tubes).forEach(([k,t])=>h+=`<tr><td>${k}</td><td>${esc(t.name)}</td><td>${t.ul}</td></tr>`);
 h+=`</table><p class="small">Rack: 4 x 5 slots (A1..D5), other slots hold empty tubes; slots ${b.bench.empty_slots.join(', ')} are empty. Tray: wells A1..A5, B1..B5. 100 pipette tips.<br>
 Limits: up to ${b.limits.tip_max_ul} uL per transfer; a tube can be drawn from while ${b.limits.tube_min_remaining_ul} uL stay in it; a well takes ${b.limits.well_max_ul} uL; waits are shortened to ${b.limits.max_wait_s} s.</p>`;$('bench').innerHTML=h;
 $('planBtn').onclick=showPlan;$('runBtn').onclick=()=>run(false);$('stopBtn').onclick=()=>j('/api/stop',{});showPlan();loadRuns();const [,s]=await j('/api/status');if(s.running){poll=setInterval(tick,1000)}else if(s.state!=='idle'){tick()}})();
</script></body></html>"""

if __name__ == "__main__":
    RUNS.mkdir(parents=True, exist_ok=True)
    print(f"Lab bench experiment runner: http://{HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
