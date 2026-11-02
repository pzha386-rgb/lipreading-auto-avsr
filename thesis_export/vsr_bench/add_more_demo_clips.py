"""Add more multi-speaker demo clips to the lip_compare_gui.
- Picks top-confidence clips for each target speaker from T1 CSV.
- Copies mp4s to videos/multispeakerdemo/
- Rebuilds T1_multispeakerdemo.csv and B2_multispeakerdemo.csv (cols: utt_id, ref, hyp, seconds)
"""
import os, shutil, pandas as pd, glob

T1_CSV = "/root/autodl-tmp/day1/conf/T1_m0v2_multispeaker_test_confA.csv"
B2_CSV = "/root/autodl-tmp/day1/conf/B2_multispeaker_test_confA.csv"
CLIP_ROOT = "/root/autodl-tmp/speaker_clips/test"      # source mp4s
GUI_VID   = "/root/autodl-tmp/vsr_bench/gui/videos/multispeakerdemo"
GUI_RES   = "/root/autodl-tmp/vsr_bench/gui/results"
os.makedirs(GUI_VID, exist_ok=True); os.makedirs(GUI_RES, exist_ok=True)

# N clips to add per speaker (on top of what's already there)
TARGETS = {
    "nathaniel_drew":   1,
    "dave2d":           1,
    "peter_mckinnon":   1,
    "jack_edwards":     1,
    "rachel_aust":      1,
    "mrwhosetheboss":   2,   # new speaker, add 2
}

def find_source_mp4(utt_id):
    """Resolve utt_id -> source mp4 path. utt_id = speaker__videoid__clip_XXXX"""
    parts = utt_id.split("__")
    if len(parts) < 3: return None
    speaker, video_id, clip_name = parts[0], parts[1], "__".join(parts[2:])
    # Try a few plausible layouts
    candidates = [
        f"{CLIP_ROOT}/{speaker}/{video_id}/{clip_name}.mp4",
        f"{CLIP_ROOT}/{speaker}/{clip_name}.mp4",
        f"{CLIP_ROOT}/{speaker}/{video_id}__{clip_name}.mp4",
    ]
    for c in candidates:
        if os.path.exists(c): return c
    # Fallback: glob
    hits = glob.glob(f"{CLIP_ROOT}/{speaker}/**/{clip_name}.mp4", recursive=True)
    return hits[0] if hits else None

# 1. Load CSVs
t1 = pd.read_csv(T1_CSV); b2 = pd.read_csv(B2_CSV)
print(f"T1: {len(t1)} rows, B2: {len(b2)} rows")

# 2. Figure out what's already in the demo folder
existing = {os.path.splitext(f)[0] for f in os.listdir(GUI_VID) if f.endswith(".mp4")}
print(f"Already have {len(existing)} clips: {sorted(existing)}")

# 3. For each target speaker, pick top-N clips by confA_logp, excluding existing
new_rows_t1, new_rows_b2 = [], []
copied, skipped = [], []
for sp, n_add in TARGETS.items():
    cand = t1[t1["id"].str.startswith(f"{sp}__")].copy()
    cand = cand[~cand["id"].isin(existing)]
    cand = cand.sort_values("confA_logp", ascending=False)
    picked = cand.head(n_add * 3)  # take more, in case source mp4 missing
    added = 0
    for _, row in picked.iterrows():
        if added >= n_add: break
        utt = row["id"]
        src = find_source_mp4(utt)
        if src is None:
            skipped.append((utt, "no source mp4"))
            continue
        dst = f"{GUI_VID}/{utt}.mp4"
        shutil.copy2(src, dst)
        copied.append((sp, utt, row["confA_logp"]))
        added += 1
    if added < n_add:
        print(f"  ⚠ {sp}: wanted {n_add}, got {added}")

print(f"\n✓ Copied {len(copied)} new clips:")
for sp, utt, lp in copied:
    print(f"  {sp:20s}  logp={lp:+.3f}  {utt}")
if skipped:
    print(f"⚠ Skipped {len(skipped)}:")
    for u, r in skipped: print(f"  {u}  ({r})")

# 4. Build final demo CSVs covering ALL mp4s in the folder (old + new)
final_utts = sorted({os.path.splitext(f)[0] for f in os.listdir(GUI_VID) if f.endswith(".mp4")})
print(f"\nFinal demo has {len(final_utts)} clips total")

def build_demo_csv(full_df, out_path):
    sub = full_df[full_df["id"].isin(final_utts)].copy()
    sub = sub.rename(columns={"id": "utt_id", "hyp": "hyp"})  # id -> utt_id
    cols = ["utt_id", "ref", "hyp"] + (["seconds"] if "seconds" in sub.columns else [])
    sub = sub[cols]
    sub.to_csv(out_path, index=False)
    return len(sub)

n_t1 = build_demo_csv(t1, f"{GUI_RES}/T1_multispeakerdemo.csv")
n_b2 = build_demo_csv(b2, f"{GUI_RES}/B2_multispeakerdemo.csv")
print(f"✓ T1_multispeakerdemo.csv: {n_t1} rows")
print(f"✓ B2_multispeakerdemo.csv: {n_b2} rows")
print("\nDone. Restart the GUI to see new clips.")
