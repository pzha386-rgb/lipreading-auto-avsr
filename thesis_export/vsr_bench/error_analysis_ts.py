# -*- coding: utf-8 -*-
"""
error_analysis_ts.py  -  what makes new speakers harder?

Compares error patterns between LRS3 (seen speakers) and Tom Scott test (new speaker):
  1. WER by sentence length
  2. Most confused word pairs (viseme-level)
  3. Insertion / deletion / substitution breakdown
  4. How LoRA changes the error pattern

Run in /root/autodl-tmp/vsr_bench:
  python error_analysis_ts.py
"""
import os, sys, re
from collections import Counter
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/day1")
from conf_utils import wer_counts, normalize, edit_distance

CONF = "/root/autodl-tmp/day1/conf"
FIGS = "/root/autodl-tmp/day1/figs"
os.makedirs(FIGS, exist_ok=True)

# Viseme groups (words that look the same on the lips)
VISEME = {
    "p": "pbm", "b": "pbm", "m": "pbm",
    "f": "fv", "v": "fv",
    "t": "tdn", "d": "tdn", "n": "tdn",
    "k": "kg", "g": "kg",
    "s": "sz", "z": "sz",
    "th": "thð", "ð": "thð",
}


def load(model, ds):
    p = f"{CONF}/{model}_{ds}_confA.csv"
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, keep_default_na=False)


def align_and_classify(ref_words, hyp_words):
    """Align words and classify each error as sub/ins/del"""
    nr, nh = len(ref_words), len(hyp_words)
    dp = [[0] * (nh + 1) for _ in range(nr + 1)]
    bt = [[None] * (nh + 1) for _ in range(nr + 1)]
    for i in range(nr + 1):
        dp[i][0] = i
    for j in range(nh + 1):
        dp[0][j] = j
    for i in range(1, nr + 1):
        for j in range(1, nh + 1):
            costs = [
                (dp[i-1][j-1] + (0 if ref_words[i-1] == hyp_words[j-1] else 1), "sub" if ref_words[i-1] != hyp_words[j-1] else "match"),
                (dp[i-1][j] + 1, "del"),
                (dp[i][j-1] + 1, "ins"),
            ]
            best = min(costs, key=lambda x: x[0])
            dp[i][j] = best[0]
            bt[i][j] = best[1]
    # traceback
    ops = []
    i, j = nr, nh
    while i > 0 or j > 0:
        if i > 0 and j > 0 and bt[i][j] in ("sub", "match"):
            if bt[i][j] == "sub":
                ops.append(("sub", ref_words[i-1], hyp_words[j-1]))
            i -= 1; j -= 1
        elif i > 0 and (j == 0 or (bt[i][j] == "del" if bt[i][j] else dp[i-1][j] < dp[i][j-1])):
            ops.append(("del", ref_words[i-1], ""))
            i -= 1
        elif j > 0:
            ops.append(("ins", "", hyp_words[j-1]))
            j -= 1
        else:
            break
    return list(reversed(ops))


def analyze(model, ds, label):
    df = load(model, ds)
    if df is None:
        return None
    sub_pairs = Counter()
    type_counts = Counter()
    wer_by_len = {}

    for _, r in df.iterrows():
        rw = normalize(r["ref"]).split()
        hw = normalize(r["hyp"]).split()
        n = len(rw)
        e, _ = wer_counts(r["ref"], r["hyp"])

        # WER by sentence length
        bucket = f"{(n // 5) * 5}-{(n // 5) * 5 + 4}" if n < 20 else "20+"
        wer_by_len.setdefault(bucket, []).append(e / max(n, 1))

        # error types
        ops = align_and_classify(rw, hw)
        for op, ref_w, hyp_w in ops:
            if op != "match":
                type_counts[op] += 1
            if op == "sub":
                sub_pairs[(ref_w, hyp_w)] += 1

    total_err = sum(type_counts.values())
    return {
        "label": label,
        "total_errors": total_err,
        "sub_%": round(100 * type_counts["sub"] / max(total_err, 1), 1),
        "del_%": round(100 * type_counts["del"] / max(total_err, 1), 1),
        "ins_%": round(100 * type_counts["ins"] / max(total_err, 1), 1),
        "top_confusions": sub_pairs.most_common(15),
        "wer_by_len": {k: round(100 * np.mean(v), 1) for k, v in sorted(wer_by_len.items())},
    }


results = []
for model, ds, label in [
    ("T1", "lrs3_300", "T1 on LRS3"),
    ("T1", "tomscott_test", "T1 on Tom Scott"),
    ("T1", "bbc", "T1 on BBC"),
    ("L15", "tomscott_test", "T1+LoRA(15min) on Tom Scott"),
    ("B2", "tomscott_test", "B2 on Tom Scott"),
    ("B2", "lrs3_300", "B2 on LRS3"),
]:
    r = analyze(model, ds, label)
    if r:
        results.append(r)

print("=" * 70)
print("ERROR TYPE BREAKDOWN")
print("=" * 70)
for r in results:
    print(f"\n{r['label']}  ({r['total_errors']} errors)")
    print(f"  Substitution: {r['sub_%']}%   Deletion: {r['del_%']}%   Insertion: {r['ins_%']}%")

print("\n" + "=" * 70)
print("TOP WORD CONFUSIONS (substitutions)")
print("=" * 70)
for r in results:
    print(f"\n{r['label']}:")
    for (ref, hyp), count in r["top_confusions"][:8]:
        print(f"  {ref:15s} -> {hyp:15s}  ({count}x)")

print("\n" + "=" * 70)
print("WER BY SENTENCE LENGTH")
print("=" * 70)
lens_df = pd.DataFrame({r["label"]: r["wer_by_len"] for r in results})
print(lens_df.to_string())
lens_df.to_csv(f"{FIGS}/error_by_length_comparison.csv")

# ---- Figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# Error type comparison
ax = axes[0]
labels_plot = [r["label"].replace(" on ", "\n") for r in results[:4]]
subs = [r["sub_%"] for r in results[:4]]
dels = [r["del_%"] for r in results[:4]]
ins = [r["ins_%"] for r in results[:4]]
x = range(len(labels_plot))
ax.bar(x, subs, label="Substitution", color="#1f77b4")
ax.bar(x, dels, bottom=subs, label="Deletion", color="#ff7f0e")
ax.bar(x, ins, bottom=[s + d for s, d in zip(subs, dels)], label="Insertion", color="#2ca02c")
ax.set_xticks(x, labels_plot, fontsize=8)
ax.set_ylabel("Error type (%)")
ax.set_title("Error type breakdown")
ax.legend()

# WER by length
ax = axes[1]
for r in results[:4]:
    keys = sorted(r["wer_by_len"].keys())
    ax.plot(keys, [r["wer_by_len"][k] for k in keys], "o-", label=r["label"].split(" on ")[0], markersize=4)
ax.set_xlabel("Sentence length (words)")
ax.set_ylabel("WER (%)")
ax.set_title("WER by sentence length")
ax.grid(alpha=0.3)
ax.legend(fontsize=8)

plt.tight_layout()
plt.savefig(f"{FIGS}/error_analysis_comparison.png", dpi=220)
print(f"\nsaved {FIGS}/error_analysis_comparison.png")
