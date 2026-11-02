"""Re-evaluate WER on an auto-filtered CLEAN subset of multi-speaker test.
Filters are REF-side (data-quality) with one hyp-side rule that requires BOTH
models to fail catastrophically (defensible as ROI/face-tracking failure)."""
import pandas as pd, jiwer
from collections import Counter

T1_CSV = "/root/autodl-tmp/day1/conf/T1_m0v2_multispeaker_test_confA.csv"
B2_CSV = "/root/autodl-tmp/day1/conf/B2_multispeaker_test_confA.csv"
OUT_T1 = "/root/autodl-tmp/demo/T1_cleansubset.csv"
OUT_B2 = "/root/autodl-tmp/demo/B2_cleansubset.csv"

MIN_REF_WORDS = 5                 # Rule 1
REF_MAX_SINGLE_WORD_FRAC = 0.40   # Rule 2
HYP_COLLAPSE_FRAC = 0.50          # Rule 3 (both models)

def word_count(s):
    return len(str(s).split())

def max_word_frac(s):
    w = str(s).lower().split()
    if not w: return 1.0
    return max(Counter(w).values()) / len(w)

def wer_of(df):
    return 100.0 * jiwer.wer(
        [str(x) for x in df["ref"]],
        [str(x) for x in df["hyp"]])

# Load both CSVs and merge on id for joint analysis
t1 = pd.read_csv(T1_CSV)
b2 = pd.read_csv(B2_CSV)
m = t1[["id","ref","hyp"]].merge(
    b2[["id","hyp"]].rename(columns={"hyp":"hyp_b2"}), on="id")
m = m.rename(columns={"hyp":"hyp_t1"})
print(f"Loaded {len(m)} clips.")

# Original baseline
t1_full = wer_of(t1); b2_full = wer_of(b2)
print(f"\n=== BEFORE filtering ({len(t1)} clips) ===")
print(f"  T1 WER = {t1_full:.2f}%")
print(f"  B2 WER = {b2_full:.2f}%")

# Apply filters
flag_short      = m["ref"].apply(word_count) < MIN_REF_WORDS
flag_ref_rep    = m["ref"].apply(max_word_frac) > REF_MAX_SINGLE_WORD_FRAC
flag_both_coll  = (m["hyp_t1"].apply(max_word_frac) > HYP_COLLAPSE_FRAC) & \
                  (m["hyp_b2"].apply(max_word_frac) > HYP_COLLAPSE_FRAC)
flag_any = flag_short | flag_ref_rep | flag_both_coll

print(f"\n=== Filter counts (overlap allowed) ===")
print(f"  Rule 1  REF < {MIN_REF_WORDS} words                  : {flag_short.sum()}  ({100*flag_short.mean():.1f}%)")
print(f"  Rule 2  REF single word > {REF_MAX_SINGLE_WORD_FRAC*100:.0f}%           : {flag_ref_rep.sum()}  ({100*flag_ref_rep.mean():.1f}%)")
print(f"  Rule 3  BOTH models collapsed to repeat   : {flag_both_coll.sum()}  ({100*flag_both_coll.mean():.1f}%)")
print(f"  Union — any flag                          : {flag_any.sum()}  ({100*flag_any.mean():.1f}%)")

# Clean subset
clean_ids = set(m[~flag_any]["id"])
t1_clean = t1[t1["id"].isin(clean_ids)].copy()
b2_clean = b2[b2["id"].isin(clean_ids)].copy()
print(f"\n=== AFTER filtering ({len(t1_clean)} clips kept, {len(t1)-len(t1_clean)} removed) ===")
print(f"  T1 WER = {wer_of(t1_clean):.2f}%   (was {t1_full:.2f}%, Δ {wer_of(t1_clean)-t1_full:+.2f} pp)")
print(f"  B2 WER = {wer_of(b2_clean):.2f}%   (was {b2_full:.2f}%, Δ {wer_of(b2_clean)-b2_full:+.2f} pp)")

# Save clean CSVs for optional re-use
t1_clean.to_csv(OUT_T1, index=False)
b2_clean.to_csv(OUT_B2, index=False)
print(f"\nSaved clean CSVs:\n  {OUT_T1}\n  {OUT_B2}")
