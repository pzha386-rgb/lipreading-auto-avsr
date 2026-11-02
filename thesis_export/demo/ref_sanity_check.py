"""Sample random clips to manually inspect REF quality."""
import pandas as pd, random
random.seed(42)

T1 = "/root/autodl-tmp/day1/conf/T1_m0v2_multispeaker_test_confA.csv"
B2 = "/root/autodl-tmp/day1/conf/B2_multispeaker_test_confA.csv"

t1 = pd.read_csv(T1); b2 = pd.read_csv(B2)
m = t1[["id","ref","hyp"]].rename(columns={"hyp":"T1"}).merge(
    b2[["id","hyp"]].rename(columns={"hyp":"B2"}), on="id")
m["speaker"] = m["id"].str.split("__").str[0]

# Sample 2-3 clips per speaker
print("=" * 85)
print("MANUAL REF-QUALITY CHECK — look at each clip's video, judge if REF is correct")
print("=" * 85)
for sp, grp in m.groupby("speaker"):
    picks = grp.sample(min(3, len(grp)), random_state=42)
    print(f"\n### {sp.upper()} ###")
    for _, r in picks.iterrows():
        print(f"\n  Clip:  {r['id']}.mp4")
        print(f"  REF:   {r['ref']}")
        print(f"  T1:    {r['T1']}")
        print(f"  B2:    {r['B2']}")

print(f"\n\nTotal {len(m)} clips in test set. Showing 3 per speaker = ~24 clips.")
print(f"Videos at: /root/autodl-tmp/vsr_bench/gui/videos/multispeakerdemo/ (if available)")
print(f"Or:        /root/autodl-tmp/speaker_clips/test/<speaker>/<video>/<clip>.mp4")
