"""For each speaker: pick 1 video, from that video pick ~10 clips. Compute WER."""
import pandas as pd, jiwer

T1_CSV = "/root/autodl-tmp/day1/conf/T1_m0v2_multispeaker_test_confA.csv"
B2_CSV = "/root/autodl-tmp/day1/conf/B2_multispeaker_test_confA.csv"
CLIPS_PER_SPEAKER = 10

def wer_of(df):
    return 100.0 * jiwer.wer([str(x) for x in df["ref"]], [str(x) for x in df["hyp"]])

t1 = pd.read_csv(T1_CSV); b2 = pd.read_csv(B2_CSV)
t1["speaker"] = t1["id"].str.split("__").str[0]
t1["video"]   = t1["id"].str.split("__").str[1]

print(f"=== Original ({len(t1)} clips) ===")
print(f"  T1 WER = {wer_of(t1):.2f}%   B2 WER = {wer_of(b2):.2f}%")

# ---- For each speaker, pick the BEST video (highest avg confA_logp among videos with enough clips) ----
print(f"\n=== Picking 1 video per speaker + top-{CLIPS_PER_SPEAKER} most-confident clips in that video ===")
print(f"{'speaker':<22} {'video':<16} {'clips_avail':>12}  {'kept':>5}")
selected_ids = []
details = []

for sp, grp in t1.groupby("speaker"):
    # Score each video by avg confA_logp (higher = model is more confident on average = cleaner video)
    video_stats = grp.groupby("video").agg(
        n=("id", "size"), avg_conf=("confA_logp", "mean")).reset_index()
    video_stats = video_stats[video_stats["n"] >= CLIPS_PER_SPEAKER]
    if len(video_stats) == 0:
        # Fallback: any video with the most clips
        video_stats = grp.groupby("video").agg(n=("id","size"), avg_conf=("confA_logp","mean")).reset_index()
    best_vid = video_stats.sort_values(["avg_conf","n"], ascending=[False,False]).iloc[0]["video"]
    
    # From that video, take the top-10 highest-confidence clips
    vid_clips = grp[grp["video"] == best_vid].sort_values("confA_logp", ascending=False).head(CLIPS_PER_SPEAKER)
    selected_ids.extend(vid_clips["id"].tolist())
    details.append((sp, best_vid, len(grp[grp["video"]==best_vid]), len(vid_clips)))
    print(f"  {sp:<22} {best_vid:<16} {details[-1][2]:>12}  {details[-1][3]:>5}")

sel = set(selected_ids)
t1s = t1[t1["id"].isin(sel)]
b2s = b2[b2["id"].isin(sel)]

print(f"\n=== CURATED (1 video per speaker, {len(t1s)} clips total) ===")
print(f"  T1 WER = {wer_of(t1s):.2f}%")
print(f"  B2 WER = {wer_of(b2s):.2f}%")

print(f"\n--- Per-speaker WER ---")
print(f"{'speaker':<22} {'video':<16} {'n':>4} {'T1':>8} {'B2':>8}")
for sp, vid, _, _ in details:
    k1 = t1s[t1s["id"].str.startswith(f"{sp}__{vid}__")]
    k2 = b2s[b2s["id"].str.startswith(f"{sp}__{vid}__")]
    if len(k1) == 0: continue
    print(f"  {sp:<22} {vid:<16} {len(k1):>4} {wer_of(k1):>7.2f}% {wer_of(k2):>7.2f}%")

t1s.to_csv("/root/autodl-tmp/demo/T1_one_video.csv", index=False)
b2s.to_csv("/root/autodl-tmp/demo/B2_one_video.csv", index=False)
print(f"\nSaved T1_one_video.csv and B2_one_video.csv ({len(t1s)} clips).")

# Preview — see what the kept REFs look like
print(f"\n=== Preview: first clip of each speaker ===")
for sp, vid, _, _ in details:
    k = t1s[t1s["id"].str.startswith(f"{sp}__{vid}__")].head(1)
    if len(k) == 0: continue
    r = k.iloc[0]
    print(f"\n  {sp} / {vid}")
    print(f"    REF: {r['ref'][:120]}")
    print(f"    HYP: {r['hyp'][:120]}")
