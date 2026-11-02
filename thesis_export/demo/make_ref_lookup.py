"""Make a human-friendly ref lookup for the 15 demo clips."""
import pandas as pd, os

T1_CSV = "/root/autodl-tmp/day1/conf/T1_m0v2_multispeaker_test_confA.csv"
GUI_VID = "/root/autodl-tmp/vsr_bench/gui/videos/multispeakerdemo"
OUT_CSV = "/root/autodl-tmp/demo/demo_clips_reference.csv"
OUT_TXT = "/root/autodl-tmp/demo/demo_clips_reference.txt"

utts = sorted({os.path.splitext(f)[0] for f in os.listdir(GUI_VID) if f.endswith(".mp4")})
t1 = pd.read_csv(T1_CSV)
sub = t1[t1["id"].isin(utts)][["id", "ref", "hyp", "confA_logp"]].copy()
sub["speaker"] = sub["id"].str.split("__").str[0]
sub = sub.sort_values(["speaker", "confA_logp"], ascending=[True, False])
sub = sub[["speaker", "id", "ref", "hyp", "confA_logp"]]
sub.to_csv(OUT_CSV, index=False)

# Also plain text for easy copy-paste on demo day
with open(OUT_TXT, "w") as f:
    f.write("DEMO CLIPS — REFERENCE TEXT (copy-paste into GUI Reference box)\n")
    f.write("=" * 70 + "\n\n")
    for sp, grp in sub.groupby("speaker"):
        f.write(f"### {sp.upper()} ###\n")
        for _, r in grp.iterrows():
            f.write(f"\n  Clip:  {r['id']}.mp4\n")
            f.write(f"  Conf:  logp={r['confA_logp']:+.3f}\n")
            f.write(f"  REF:   {r['ref']}\n")
            f.write(f"  T1:    {r['hyp']}\n")
        f.write("\n" + "-" * 70 + "\n\n")

print(f"✓ CSV:  {OUT_CSV}  ({len(sub)} rows)")
print(f"✓ TXT:  {OUT_TXT}")
print("\n=== Preview ===")
print(sub.to_string(index=False, max_colwidth=50))
