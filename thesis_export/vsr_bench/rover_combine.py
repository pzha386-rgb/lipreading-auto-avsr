# -*- coding: utf-8 -*-
"""
rover_combine.py  -  combine T1 (Auto-AVSR) and B2 (BRAVEn) outputs

Two strategies:
  sentence: pick whichever model's full hypothesis has higher confidence
  word:     align the two outputs word-by-word, vote at each position

Run in /root/autodl-tmp/vsr_bench:
  python rover_combine.py
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/day1")
from conf_utils import wer_counts, normalize, edit_distance

CONF = "/root/autodl-tmp/day1/conf"
FIGS = "/root/autodl-tmp/day1/figs"
os.makedirs(FIGS, exist_ok=True)

SETS = [("tomscott_test", "Tom Scott test"), ("bbc", "BBC"), ("lrs3_300", "LRS3")]


def load(model, ds):
    p = f"{CONF}/{model}_{ds}_confA.csv"
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, keep_default_na=False)


def align_words(a_words, b_words):
    """DP alignment of two word sequences, returns list of (a_word_or_None, b_word_or_None)"""
    na, nb = len(a_words), len(b_words)
    dp = [[0] * (nb + 1) for _ in range(na + 1)]
    for i in range(na + 1):
        dp[i][0] = i
    for j in range(nb + 1):
        dp[0][j] = j
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1,
                          dp[i-1][j-1] + (0 if a_words[i-1] == b_words[j-1] else 1))
    # traceback
    pairs = []
    i, j = na, nb
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (0 if a_words[i-1] == b_words[j-1] else 1):
            pairs.append((a_words[i-1], b_words[j-1]))
            i -= 1; j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            pairs.append((a_words[i-1], None))
            i -= 1
        else:
            pairs.append((None, b_words[j-1]))
            j -= 1
    return list(reversed(pairs))


def rover_word(hyp_a, hyp_b, conf_a, conf_b):
    """Word-level ROVER: align, then at each position pick the more confident model's word"""
    wa = normalize(hyp_a).split()
    wb = normalize(hyp_b).split()
    if not wa and not wb:
        return ""
    if not wa:
        return normalize(hyp_b)
    if not wb:
        return normalize(hyp_a)
    pairs = align_words(wa, wb)
    out = []
    for a, b in pairs:
        if a == b:
            out.append(a)
        elif a is None:
            # b has an extra word; include if B is more confident, skip if A is
            if conf_b > conf_a:
                out.append(b)
        elif b is None:
            if conf_a > conf_b:
                out.append(a)
        else:
            # substitution: pick the word from the more confident model
            out.append(a if conf_a >= conf_b else b)
    return " ".join(out)


def rover_sentence(hyp_a, hyp_b, conf_a, conf_b):
    """Sentence-level: pick the hypothesis from the model with higher confidence"""
    return normalize(hyp_a) if conf_a >= conf_b else normalize(hyp_b)


rows = []
for ds, label in SETS:
    t1 = load("T1", ds)
    b2 = load("B2", ds)
    if t1 is None or b2 is None:
        print(f"skip {ds}: missing data")
        continue

    m = t1[["id", "ref", "hyp", "confA_logp"]].merge(
        b2[["id", "hyp", "confA_logp"]].rename(columns={"hyp": "hyp_b", "confA_logp": "confA_logp_b"}),
        on="id", how="inner")

    for method, fn in [("T1 only", lambda r: r["hyp"]),
                       ("B2 only", lambda r: r["hyp_b"]),
                       ("sentence ROVER", lambda r: rover_sentence(r["hyp"], r["hyp_b"], r["confA_logp"], r["confA_logp_b"])),
                       ("word ROVER", lambda r: rover_word(r["hyp"], r["hyp_b"], r["confA_logp"], r["confA_logp_b"]))]:
        errs, nw = 0, 0
        for _, r in m.iterrows():
            combined = fn(r)
            e, n = wer_counts(r["ref"], combined)
            errs += e; nw += n
        wer = 100 * errs / max(nw, 1)
        rows.append({"method": method, "dataset": label, "WER": round(wer, 2)})

    # also: oracle (pick whichever hypothesis has lower WER for each sentence)
    errs_o, nw_o = 0, 0
    for _, r in m.iterrows():
        e1, n = wer_counts(r["ref"], r["hyp"])
        e2, _ = wer_counts(r["ref"], r["hyp_b"])
        errs_o += min(e1, e2); nw_o += n
    rows.append({"method": "oracle (upper bound)", "dataset": label, "WER": round(100 * errs_o / max(nw_o, 1), 2)})

res = pd.DataFrame(rows)
pivot = res.pivot(index="method", columns="dataset", values="WER")
pivot = pivot.reindex(["T1 only", "B2 only", "sentence ROVER", "word ROVER", "oracle (upper bound)"])
pivot = pivot[[c for _, c in SETS if c in pivot.columns]]
print(pivot.to_string())
pivot.to_csv(f"{FIGS}/rover_results.csv")

# count how often word ROVER differs from both T1 and B2
for ds, label in SETS:
    t1, b2 = load("T1", ds), load("B2", ds)
    if t1 is None or b2 is None:
        continue
    m = t1[["id", "ref", "hyp", "confA_logp"]].merge(
        b2[["id", "hyp", "confA_logp"]].rename(columns={"hyp": "hyp_b", "confA_logp": "confA_logp_b"}),
        on="id", how="inner")
    agree = sum(normalize(a) == normalize(b) for a, b in zip(m["hyp"], m["hyp_b"]))
    t1_pick = sum(r["confA_logp"] >= r["confA_logp_b"] for _, r in m.iterrows())
    print(f"\n{label}: {len(m)} sentences, {agree} agree ({100*agree/len(m):.0f}%), "
          f"T1 picked {t1_pick} times ({100*t1_pick/len(m):.0f}%)")

print(f"\nsaved {FIGS}/rover_results.csv")
