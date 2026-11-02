"""HTML5 native audit tool — bypasses Gradio video issues."""
import os, json, random, glob, shutil, pandas as pd, jiwer
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn

random.seed(42)
T1_CSV = "/root/autodl-tmp/day1/conf/T1_m0v2_multispeaker_test_confA.csv"
B2_CSV = "/root/autodl-tmp/day1/conf/B2_multispeaker_test_confA.csv"
CLIP_ROOT = "/root/autodl-tmp/speaker_clips/test"
STAGE_DIR = "/root/autodl-tmp/demo/audit_videos"
JUDGE_FILE = "/root/autodl-tmp/demo/audit_judgments.json"
CAND_PER_SPEAKER = 15
TARGET_CORRECT = 8

t1 = pd.read_csv(T1_CSV)
t1["speaker"] = t1["id"].str.split("__").str[0]

# Build candidate list (reuse existing staged videos if present)
candidates = {}
for sp, grp in t1.groupby("speaker"):
    sorted_grp = grp.sort_values("confA_logp", ascending=False).head(40)
    picked = sorted_grp.sample(min(CAND_PER_SPEAKER, len(sorted_grp)), random_state=42)
    sp_clips = []
    for _, r in picked.iterrows():
        dst = f"{STAGE_DIR}/{r['id']}.mp4"
        if os.path.exists(dst):
            sp_clips.append({"id": r["id"], "ref": r["ref"]})
    if sp_clips:
        candidates[sp] = sp_clips

judgments = json.load(open(JUDGE_FILE)) if os.path.exists(JUDGE_FILE) else {}

app = FastAPI()

@app.get("/video/{uid}")
def get_video(uid: str):
    p = f"{STAGE_DIR}/{uid}.mp4"
    if not os.path.exists(p): return JSONResponse({"error": "not found"}, 404)
    return FileResponse(p, media_type="video/mp4")

@app.get("/state")
def get_state():
    speakers = sorted(candidates.keys())
    for sp in speakers:
        done = sum(1 for c in candidates[sp] if judgments.get(c["id"]) == "correct")
        if done >= TARGET_CORRECT: continue
        for c in candidates[sp]:
            if c["id"] not in judgments:
                return {"done": False, "speaker": sp, "done_count": done,
                        "target": TARGET_CORRECT, "id": c["id"], "ref": c["ref"],
                        "total_judged": len(judgments)}
    return {"done": True, "total_judged": len(judgments)}

@app.post("/judge/{uid}/{verdict}")
def judge(uid: str, verdict: str):
    judgments[uid] = verdict
    json.dump(judgments, open(JUDGE_FILE, "w"), indent=2)
    return get_state()

@app.get("/wer")
def wer():
    good = {uid for uid, v in judgments.items() if v == "correct"}
    if not good: return {"msg": "No clips marked correct yet"}
    b2 = pd.read_csv(B2_CSV)
    t1c = t1[t1["id"].isin(good)]
    b2c = b2[b2["id"].isin(good)]
    def w(d): return 100 * jiwer.wer([str(x) for x in d["ref"]], [str(x) for x in d["hyp"]])
    out = {"n_clips": len(t1c), "T1_WER": round(w(t1c), 2), "B2_WER": round(w(b2c), 2),
           "per_speaker": {}}
    for sp in sorted(t1c["id"].str.split("__").str[0].unique()):
        k1 = t1c[t1c["id"].str.startswith(sp+"__")]
        k2 = b2c[b2c["id"].str.startswith(sp+"__")]
        out["per_speaker"][sp] = {"n": len(k1), "T1": round(w(k1), 2), "B2": round(w(k2), 2)}
    return out

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html><html><head><title>REF Audit</title>
<style>
  body{font-family:system-ui;max-width:1100px;margin:20px auto;padding:0 20px}
  .row{display:flex;gap:20px}
  .col{flex:1}
  video{width:100%;max-height:400px;background:#000;border-radius:8px}
  .ref{background:#f0f4ff;border:2px solid #c0d0f0;padding:16px;border-radius:8px;font-size:18px;line-height:1.5;min-height:120px}
  button{font-size:16px;padding:14px 20px;margin:5px;border-radius:8px;border:0;cursor:pointer;font-weight:bold}
  .ok{background:#22c55e;color:#fff}
  .bad{background:#ef4444;color:#fff}
  .skip{background:#9ca3af;color:#fff}
  .compute{background:#2563eb;color:#fff;font-size:14px;padding:10px 16px}
  #hdr{font-size:20px;font-weight:bold;margin:16px 0;color:#1a2332}
  #result{background:#f9fafb;padding:16px;border-radius:8px;margin-top:16px;font-family:monospace;white-space:pre-wrap}
</style></head><body>
<h1>REF Quality Audit</h1>
<p>Watch each clip. Does the REF text match what the speaker actually says?</p>
<div id="hdr">Loading...</div>
<div class="row">
  <div class="col"><video id="vid" controls autoplay></video></div>
  <div class="col">
    <div class="ref" id="ref"></div>
    <div style="margin-top:12px">
      <button class="ok" onclick="judge('correct')">✅ REF matches audio</button>
      <button class="bad" onclick="judge('wrong')">❌ REF wrong</button>
    </div>
    <div><button class="skip" onclick="skipSpeaker()">⏭ Skip this speaker</button></div>
  </div>
</div>
<hr><button class="compute" onclick="computeWER()">📊 Compute WER on verified subset</button>
<div id="result"></div>

<script>
let currentId = null, currentSpeaker = null;
async function load() {
  const s = await (await fetch('/state')).json();
  if (s.done) { document.getElementById('hdr').textContent = '✅ All done! Click Compute WER below.'; document.getElementById('vid').src=''; document.getElementById('ref').textContent=''; return; }
  currentId = s.id; currentSpeaker = s.speaker;
  document.getElementById('hdr').textContent = `${s.speaker}  (${s.done_count}/${s.target} correct)  —  total judged: ${s.total_judged}`;
  document.getElementById('ref').textContent = s.ref;
  document.getElementById('vid').src = '/video/' + s.id;
}
async function judge(v) { if(!currentId) return; await fetch('/judge/' + currentId + '/' + v, {method:'POST'}); load(); }
async function skipSpeaker() {
  // Mark all remaining clips of this speaker as 'skipped' so state() advances
  const r = await fetch('/state'); const s = await r.json();
  if (!s.speaker) return;
  // Simplest: mark current as 'wrong', then until speaker changes mark the rest wrong too
  // Easier approach: just keep clicking wrong — but let's add a bulk skip
  for (let i=0; i<20; i++) {
    const cur = await (await fetch('/state')).json();
    if (cur.done || cur.speaker !== currentSpeaker) break;
    await fetch('/judge/' + cur.id + '/wrong', {method:'POST'});
  }
  load();
}
async function computeWER() {
  const r = await (await fetch('/wer')).json();
  if (r.msg) { document.getElementById('result').textContent = r.msg; return; }
  let s = `Clips verified: ${r.n_clips}\n\nT1 WER = ${r.T1_WER}%\nB2 WER = ${r.B2_WER}%\n\n--- Per-speaker ---\n`;
  for (const [sp, v] of Object.entries(r.per_speaker)) {
    s += `${sp.padEnd(22)} n=${v.n}  T1=${v.T1}%  B2=${v.B2}%\n`;
  }
  document.getElementById('result').textContent = s;
}
load();
</script></body></html>
    """

uvicorn.run(app, host="0.0.0.0", port=6006)
