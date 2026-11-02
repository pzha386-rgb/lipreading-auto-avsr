"""Manual REF-audit tool: watch each clip, mark REF correct/incorrect.
Target: 8 'correct' clips per speaker. Shows up to 15 candidates per speaker."""
import os, json, shutil, random, glob, pandas as pd, gradio as gr, jiwer

random.seed(42)
T1_CSV = "/root/autodl-tmp/day1/conf/T1_m0v2_multispeaker_test_confA.csv"
B2_CSV = "/root/autodl-tmp/day1/conf/B2_multispeaker_test_confA.csv"
CLIP_ROOT = "/root/autodl-tmp/speaker_clips/test"
STAGE_DIR = "/root/autodl-tmp/demo/audit_videos"
JUDGE_FILE = "/root/autodl-tmp/demo/audit_judgments.json"

CANDIDATES_PER_SPEAKER = 15
TARGET_CORRECT_PER_SPEAKER = 8

os.makedirs(STAGE_DIR, exist_ok=True)

# ---- Pick candidates ----
t1 = pd.read_csv(T1_CSV)
t1["speaker"] = t1["id"].str.split("__").str[0]

def find_src(utt):
    parts = utt.split("__")
    sp, vid = parts[0], parts[1]
    clip = "__".join(parts[2:])
    for p in [f"{CLIP_ROOT}/{sp}/{vid}/{clip}.mp4",
              *glob.glob(f"{CLIP_ROOT}/{sp}/**/{clip}.mp4", recursive=True)]:
        if os.path.exists(p): return p
    return None

print("Preparing candidate videos...")
candidates = {}
for sp, grp in t1.groupby("speaker"):
    sorted_grp = grp.sort_values("confA_logp", ascending=False).head(40)
    picked = sorted_grp.sample(min(CANDIDATES_PER_SPEAKER, len(sorted_grp)), random_state=42)
    speaker_clips = []
    for _, r in picked.iterrows():
        src = find_src(r["id"])
        if src is None: continue
        dst = f"{STAGE_DIR}/{r['id']}.mp4"
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
        speaker_clips.append({"id": r["id"], "ref": r["ref"], "video": dst})
    candidates[sp] = speaker_clips
    print(f"  {sp:<22} {len(speaker_clips)} clips staged")

# Load existing judgments
if os.path.exists(JUDGE_FILE):
    judgments = json.load(open(JUDGE_FILE))
    print(f"Resumed with {len(judgments)} existing judgments")
else:
    judgments = {}

speakers = sorted(candidates.keys())
state = {"speaker_idx": 0, "clip_idx": 0}

# ---- Resume logic: skip speakers already done ----
def speaker_done_count(sp):
    return sum(1 for c in candidates[sp] if judgments.get(c["id"]) == "correct")

while state["speaker_idx"] < len(speakers) and \
      speaker_done_count(speakers[state["speaker_idx"]]) >= TARGET_CORRECT_PER_SPEAKER:
    state["speaker_idx"] += 1

def current():
    if state["speaker_idx"] >= len(speakers):
        return None, None, "✅ ALL DONE — click 'Compute WER' at the bottom.", ""
    sp = speakers[state["speaker_idx"]]
    clips = candidates[sp]
    while state["clip_idx"] < len(clips) and clips[state["clip_idx"]]["id"] in judgments:
        state["clip_idx"] += 1
    if state["clip_idx"] >= len(clips):
        return None, None, f"⚠ Speaker {sp}: ran out of 15 candidates ({speaker_done_count(sp)} correct). Click Next speaker.", ""
    c = clips[state["clip_idx"]]
    done = speaker_done_count(sp)
    header = f"### {sp}  ({done}/{TARGET_CORRECT_PER_SPEAKER} correct)  —  clip {state['clip_idx']+1}/{len(clips)}"
    return c["video"], c["ref"], header, c["id"]

def judge(verdict, cur_id):
    if cur_id:
        judgments[cur_id] = verdict
        json.dump(judgments, open(JUDGE_FILE, "w"), indent=2)
    sp = speakers[state["speaker_idx"]] if state["speaker_idx"] < len(speakers) else None
    if sp and speaker_done_count(sp) >= TARGET_CORRECT_PER_SPEAKER:
        state["speaker_idx"] += 1
        state["clip_idx"] = 0
        while state["speaker_idx"] < len(speakers) and \
              speaker_done_count(speakers[state["speaker_idx"]]) >= TARGET_CORRECT_PER_SPEAKER:
            state["speaker_idx"] += 1
    else:
        state["clip_idx"] += 1
    return current()

def skip_speaker(cur_id):
    state["speaker_idx"] += 1
    state["clip_idx"] = 0
    return current()

def compute_wer():
    good = {uid for uid, v in judgments.items() if v == "correct"}
    if not good:
        return "No clips marked correct yet."
    b2 = pd.read_csv(B2_CSV)
    t1c = t1[t1["id"].isin(good)]
    b2c = b2[b2["id"].isin(good)]
    def wer(d): return 100 * jiwer.wer([str(x) for x in d["ref"]], [str(x) for x in d["hyp"]])
    out = f"## Results on human-verified subset ({len(t1c)} clips)\n\n"
    out += f"- **T1 WER = {wer(t1c):.2f}%**\n"
    out += f"- **B2 WER = {wer(b2c):.2f}%**\n\n"
    out += f"### Per-speaker breakdown\n\n"
    for sp in sorted(t1c["id"].str.split("__").str[0].unique()):
        k1 = t1c[t1c["id"].str.startswith(sp+"__")]
        k2 = b2c[b2c["id"].str.startswith(sp+"__")]
        out += f"- {sp}: n={len(k1)}, T1={wer(k1):.1f}%, B2={wer(k2):.1f}%\n"
    return out

with gr.Blocks(title="REF Audit", theme=gr.themes.Soft()) as app:
    gr.Markdown("# REF Quality Audit\nFor each clip: watch the video, read the REF, decide if REF matches what the speaker actually says.")
    header_md = gr.Markdown("")
    cur_id = gr.Textbox(visible=False)
    with gr.Row():
        with gr.Column():
            vid = gr.Video(label="Clip", autoplay=True, height=360)
        with gr.Column():
            ref_box = gr.Textbox(label="REF (YouTube caption)", lines=4, interactive=False)
            with gr.Row():
                btn_ok  = gr.Button("✅ REF matches audio", variant="primary")
                btn_bad = gr.Button("❌ REF wrong / mismatched")
            btn_skip = gr.Button("⏭ Skip this speaker (not enough good clips)")
    with gr.Row():
        btn_compute = gr.Button("📊 Compute WER on verified subset", variant="secondary")
        result = gr.Markdown("")

    def _init(): return current()
    app.load(_init, None, [vid, ref_box, header_md, cur_id])

    btn_ok.click(lambda cid: judge("correct", cid), cur_id, [vid, ref_box, header_md, cur_id])
    btn_bad.click(lambda cid: judge("wrong", cid), cur_id, [vid, ref_box, header_md, cur_id])
    btn_skip.click(skip_speaker, cur_id, [vid, ref_box, header_md, cur_id])
    btn_compute.click(compute_wer, None, result)

app.launch(server_name="0.0.0.0", server_port=6006, allowed_paths=["/root/autodl-tmp/demo/audit_videos"])
