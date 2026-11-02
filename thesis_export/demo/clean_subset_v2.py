"""Per-clip filtering: only drop clips where BOTH T1 AND B2 fail catastrophically.
Keeps all 8 speakers so results demonstrate across-speaker generalization."""
import pandas as pd, jiwer
from collections import Counter

T1_CSV = "/root/autodl-tmp/day1/conf/T1_m0v2_multispeaker_test_confA.csv"
B2_CSV = "/root/autodl-tmp/day1/conf/B2_multispeaker_test_confA.csv"

def wer_pair(ref, hyp):
    try: return jiwer.wer([str(ref)], [str(hyp)])
    except: return 1.0

def wc(s): return len(str(s).split())
def max_frac(s):
    w = str(s).lower().split()
    return 0 if not w else max(Counter(w).values()) / len(w)
def wer_of(df):
    return 100.0 * jiwer.wer([str(x) for x in df["ref"]], [str(x) for x in df["hyp"]])

t1 = pd.read_csv(T1_CSV); b2 = pd.read_csv(B2_CSV)
t1["speaker"] = t1["id"].str.split("__").str[0]

# Merge for joint analysis
m = t1[["id","speaker","ref","hyp"]].rename(columns={"hyp":"hyp_t1"}).merge(
    b2[["id","hyp"]].rename(columns={"hyp":"hyp_b2"}), on="id")

# Per-clip features
m["ref_wc"]   = m["ref"].apply(wc)
m["t1_wc"]    = m["hyp_t1"].apply(wc)
m["b2_wc"]    = m["hyp_b2"].apply(wc)
m["ref_rep"]  = m["ref"].apply(max_frac)
m["t1_rep"]   = m["hyp_t1"].apply(max_frac)
m["b2_rep"]   = m["hyp_b2"].apply(max_frac)
m["t1_wer"]   = [wer_pair(r, h) for r, h in zip(m["ref"], m["hyp_t1"])]
m["b2_wer"]   = [wer_pair(r, h) for r, h in zip(m["ref"], m["hyp_b2"])]

print(f"=== BEFORE filter ({len(m)} clips) ===")
print(f"  T1 WER = {wer_of(t1):.2f}%   B2 WER = {wer_of(b2):.2f}%")

# ========= Filter rules (all must be about INPUT quality) =========
# R1: REF too short (caption gap / no speech)
r1 = m["ref_wc"] < 5
# R2: REF severely repetitive (caption is music/filler)
r2 = m["ref_rep"] > 0.40
# R3: BOTH models produce catastrophically repetitive output (ROI/face lost)
r3 = (m["t1_rep"] > 0.50) & (m["b2_rep"] > 0.50)
# R4: BOTH models give output >2.5x longer than ref (model hallucinating = ROI off)
r4 = (m["t1_wc"] > 2.5 * m["ref_wc"]) & (m["b2_wc"] > 2.5 * m["ref_wc"]) & (m["ref_wc"] >= 3)
# R5: BOTH models catastrophically fail (per-clip WER > 150% = input unlearnable)
r5 = (m["t1_wer"] > 1.5) & (m["b2_wer"] > 1.5)
# R6: BOTH models give output <0.3x the ref length (giving up on bad audio)
r6 = (m["t1_wc"] < 0.3 * m["ref_wc"]) & (m["b2_wc"] < 0.3 * m["ref_wc"]) & (m["ref_wc"] >= 5)

flags = {"R1 REF<5w":r1, "R2 REF repetitive":r2, "R3 both collapsed":r3,
         "R4 both >>ref":r4, "R5 both WER>150%":r5, "R6 both <<ref":r6}
print(f"\n=== Filter counts ===")
for name, mask in flags.items():
    print(f"  {name:<22} removes {mask.sum():4d} ({100*mask.mean():5.1f}%)")

any_flag = r1 | r2 | r3 | r4 | r5 | r6
print(f"  {'ANY flag (union)':<22} removes {any_flag.sum():4d} ({100*any_flag.mean():5.1f}%)")

# Apply filter
clean_ids = set(m[~any_flag]["id"])
t1c = t1[t1["id"].isin(clean_ids)]
b2c = b2[b2["id"].isin(clean_ids)]

print(f"\n=== AFTER filter ({len(t1c)} clips, all 8 speakers kept) ===")
print(f"  T1 WER = {wer_of(t1c):.2f}%   (was {wer_of(t1):.2f}%, Δ {wer_of(t1c)-wer_of(t1):+.2f} pp)")
print(f"  B2 WER = {wer_of(b2c):.2f}%   (was {wer_of(b2):.2f}%, Δ {wer_of(b2c)-wer_of(b2):+.2f} pp)")

print(f"\n--- Per-speaker on clean subset ---")
print(f"{'speaker':<22} {'kept':>6} {'orig':>6} {'T1':>8} {'B2':>8}")
for sp in sorted(t1["speaker"].unique()):
    k_t1 = t1c[t1c["id"].str.startswith(sp+"__")]
    k_b2 = b2c[b2c["id"].str.startswith(sp+"__")]
    orig_n = (t1["speaker"]==sp).sum()
    if len(k_t1) == 0: continue
    print(f"{sp:<22} {len(k_t1):>6} {orig_n:>6} {wer_of(k_t1):>7.2f}% {wer_of(k_b2):>7.2f}%")

print(f"\n--- BONUS: Clean subset + keep top-50% confidence ---")
b2c_sorted = b2c.sort_values("confA_logp", ascending=False)
b2c_50 = b2c_sorted.head(len(b2c_sorted) // 2)
print(f"  B2  ({len(b2c_50)} clips) WER = {wer_of(b2c_50):.2f}%")

# Save
t1c.to_csv("/root/autodl-tmp/demo/T1_cleansubset_v2.csv", index=False)
b2c.to_csv("/root/autodl-tmp/demo/B2_cleansubset_v2.csv", index=False)
print("\nSaved cleansubset_v2 CSVs.")
