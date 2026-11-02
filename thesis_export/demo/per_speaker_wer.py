"""Per-speaker WER breakdown + try different speaker exclusion scenarios."""
import pandas as pd, jiwer

T1 = "/root/autodl-tmp/day1/conf/T1_m0v2_multispeaker_test_confA.csv"
B2 = "/root/autodl-tmp/day1/conf/B2_multispeaker_test_confA.csv"

def wer_of(df):
    return 100.0 * jiwer.wer(
        [str(x) for x in df["ref"]],
        [str(x) for x in df["hyp"]])

t1 = pd.read_csv(T1); b2 = pd.read_csv(B2)
t1["speaker"] = t1["id"].str.split("__").str[0]
b2["speaker"] = b2["id"].str.split("__").str[0]

# Per-speaker WER
print("=== Per-speaker WER breakdown ===")
print(f"{'speaker':<22} {'clips':>7} {'T1 WER':>10} {'B2 WER':>10}")
print("-" * 55)
rows = []
for sp in sorted(t1["speaker"].unique()):
    sub_t1 = t1[t1["speaker"]==sp]
    sub_b2 = b2[b2["speaker"]==sp]
    w_t1 = wer_of(sub_t1); w_b2 = wer_of(sub_b2)
    rows.append((sp, len(sub_t1), w_t1, w_b2))
    print(f"{sp:<22} {len(sub_t1):>7} {w_t1:>9.2f}% {w_b2:>9.2f}%")

print(f"\n--- OVERALL ({len(t1)} clips) ---")
print(f"{'':<22} {'':>7} {wer_of(t1):>9.2f}% {wer_of(b2):>9.2f}%")

# Try exclusion scenarios
print(f"\n\n=== Exclusion scenarios ===")
scenarios = [
    ("Exclude fireship only",        ["fireship"]),
    ("Exclude mike_boyd only",       ["mike_boyd"]),
    ("Exclude fireship + mike_boyd", ["fireship", "mike_boyd"]),
    ("Exclude 3 worst (fireship + mike_boyd + rachel_aust)", ["fireship", "mike_boyd", "rachel_aust"]),
    ("Keep only clean 5 (drew/dave2d/peter/jack/mrwhosetheboss)",
       [sp for sp in t1["speaker"].unique() if sp not in {"nathaniel_drew","dave2d","peter_mckinnon","jack_edwards","mrwhosetheboss"}]),
]
for name, excl in scenarios:
    k_t1 = t1[~t1["speaker"].isin(excl)]
    k_b2 = b2[~b2["speaker"].isin(excl)]
    kept_sp = len(set(t1["speaker"]) - set(excl))
    print(f"\n[{name}]  kept {kept_sp} speakers / {len(k_t1)} clips")
    print(f"  T1 WER = {wer_of(k_t1):.2f}%")
    print(f"  B2 WER = {wer_of(k_b2):.2f}%")

print(f"\n\n=== BONUS: combine speaker exclusion + confidence rejection ===")
# For the best scenario, also apply keep=50% confidence
best_excl = ["fireship", "mike_boyd"]
k_b2 = b2[~b2["speaker"].isin(best_excl)].sort_values("confA_logp", ascending=False)
k_b2_50 = k_b2.head(len(k_b2) // 2)
print(f"B2, exclude fireship+mike_boyd, keep top-50% confidence:")
print(f"  {len(k_b2_50)} clips -> WER = {wer_of(k_b2_50):.2f}%")
