# -*- coding: utf-8 -*-
"""
错误分析（只用现有结果，CPU，约 1-2 分钟）
  1. 错误类型：替换 / 漏词 / 多词 各占多少
  2. 句子长度对 WER 的影响
  3. 词的长度对识别率的影响（短的虚词是否更容易漏）
  4. 最常见的混淆词对，以及其中"口型相同"的比例
  5. 空输出统计
  6. 含数字的句子是否更难

输出：/root/autodl-tmp/figures/fig8_*.png, fig9_*.png, error_*.csv
运行：python error_analysis.py
"""
import os
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import ROOT

R = f"{ROOT}/results_v2"
OUT = f"{ROOT}/figures"
os.makedirs(OUT, exist_ok=True)

REPS = ["S4", "S0", "T1", "T4", "T3", "B4", "B5", "U4", "U4g"]       # 代表模型（从弱到强）
NAME = {"S4": "Ma2022 (sup.)", "S0": "Auto-AVSR 438h (sup.)", "T1": "Auto-AVSR 3291h (sup.)",
        "T4": "RAVEn low +ST", "T3": "BRAVEn low +ST", "B4": "BRAVEn high +ST",
        "B5": "BRAVEn high +ST +LM", "U4": "USR2 Huge", "U4g": "USR2 Huge (greedy)"}
NUMBER_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
                   "fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty sixty seventy "
                   "eighty ninety hundred thousand million billion".split())
# 粗略的"口型类"（按首字母近似，只用于统计口型混淆的大致比例）
VISEME = {**{c: "bilabial" for c in "pbm"}, **{c: "labiodental" for c in "fv"},
          **{c: "alveolar" for c in "tdnszl"}, **{c: "velar" for c in "kgcq"}, **{c: "round" for c in "wo"}}


def align(r, h):
    """返回 [(op, ref_word, hyp_word)]，op ∈ ok / sub / del / ins"""
    n, m = len(r), len(h)
    d = np.zeros((n + 1, m + 1), dtype=np.int32)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (r[i - 1] != h[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (r[i - 1] != h[j - 1]):
            ops.append(("ok" if r[i - 1] == h[j - 1] else "sub", r[i - 1], h[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            ops.append(("del", r[i - 1], None))
            i -= 1
        else:
            ops.append(("ins", None, h[j - 1]))
            j -= 1
    return ops[::-1]


def load(mid, ds):
    p = f"{R}/{mid}_{ds}.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p).fillna("")
    return df


def analyse(ds):
    types, by_len, by_wlen, empties, num_rows = [], [], [], [], []
    confusions = Counter()
    for mid in REPS:
        df = load(mid, ds)
        if df is None:
            continue
        cnt = Counter()
        wl = Counter()
        wl_ok = Counter()
        for _, row in df.iterrows():
            r, h = str(row.ref_norm).split(), str(row.hyp_norm).split()
            ops = align(r, h)
            c = Counter(o for o, _, _ in ops)
            cnt.update(c)
            n = len(r)
            err = c["sub"] + c["del"] + c["ins"]
            bucket = "1-3" if n <= 3 else "4-6" if n <= 6 else "7-10" if n <= 10 else "11-15" if n <= 15 else "16+"
            by_len.append({"id": mid, "bucket": bucket, "errors": err, "n": n})
            has_num = any(w in NUMBER_WORDS for w in r)
            num_rows.append({"id": mid, "has_number": has_num, "errors": err, "n": n})
            if not h:
                empties.append({"id": mid, "utt_id": row.utt_id, "ref_len": n, "ref": row.ref_norm})
            for o, rw, hw in ops:
                if rw is not None:
                    k = "1-2" if len(rw) <= 2 else "3-4" if len(rw) <= 4 else "5-7" if len(rw) <= 7 else "8+"
                    wl[k] += 1
                    wl_ok[k] += (o == "ok")
                if o == "sub" and mid in ("T1", "T3", "B4", "U4"):
                    confusions[(rw, hw)] += 1
        N = sum(cnt[k] for k in ("ok", "sub", "del"))
        types.append({"id": mid, "WER": 100 * (cnt["sub"] + cnt["del"] + cnt["ins"]) / N,
                      "substitution": 100 * cnt["sub"] / N, "deletion": 100 * cnt["del"] / N,
                      "insertion": 100 * cnt["ins"] / N})
        for k in wl:
            by_wlen.append({"id": mid, "word_len": k, "recall": 100 * wl_ok[k] / wl[k], "count": wl[k]})
    return (pd.DataFrame(types), pd.DataFrame(by_len), pd.DataFrame(by_wlen),
            pd.DataFrame(empties), pd.DataFrame(num_rows), confusions)


pd.set_option("display.width", 200)
for ds in ["lrs3", "bbc"]:
    print(f"\n################ {ds} ################")
    types, by_len, by_wlen, empties, nums, conf = analyse(ds)
    if types.empty:
        continue

    # 1. 错误类型
    types = types.round(2)
    types.to_csv(f"{OUT}/error_types_{ds}.csv", index=False)
    print("\n===== 1. 错误类型（占参考词数的百分比）=====")
    print(types.to_string(index=False))

    # 2. 句子长度
    L = by_len.groupby(["id", "bucket"]).agg(errors=("errors", "sum"), n=("n", "sum"), clips=("n", "size"))
    L["WER"] = 100 * L.errors / L.n
    order = ["1-3", "4-6", "7-10", "11-15", "16+"]
    pv = L["WER"].unstack().reindex(columns=order).reindex([m for m in REPS if m in L.index.get_level_values(0)])
    pv.round(1).to_csv(f"{OUT}/error_by_length_{ds}.csv")
    clips = L["clips"].unstack().iloc[0].reindex(order)
    print("\n===== 2. 句子长度 vs WER（列 = 参考句子的词数）=====")
    print("片段数:", dict(clips))
    print(pv.round(1).to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    for mid in pv.index:
        ax.plot(order, pv.loc[mid], "-o", ms=4, label=NAME[mid])
    ax.set_xlabel("Sentence length (words)")
    ax.set_ylabel(f"WER on {ds.upper()} (%)")
    ax.set_title("Short utterances are the hardest")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig8_length_{ds}.png", dpi=300)
    plt.close(fig)

    # 3. 词长度
    W = by_wlen.pivot(index="id", columns="word_len", values="recall").reindex(columns=["1-2", "3-4", "5-7", "8+"])
    W = W.reindex([m for m in REPS if m in W.index])
    W.round(1).to_csv(f"{OUT}/error_by_wordlen_{ds}.csv")
    print("\n===== 3. 词的长度 vs 识别正确率 %（列 = 字母数）=====")
    print(W.round(1).to_string())

    # 4. 混淆词对
    top = conf.most_common(25)
    same = sum(c for (a, b), c in conf.items() if a and b and VISEME.get(a[0]) and VISEME.get(a[0]) == VISEME.get(b[0]))
    tot = sum(conf.values())
    pd.DataFrame([{"ref": a, "hyp": b, "count": c} for (a, b), c in top]).to_csv(f"{OUT}/error_confusions_{ds}.csv", index=False)
    print(f"\n===== 4. 最常见的替换（T1 T3 B4 U4 合计）=====")
    print(", ".join(f"{a}→{b}({c})" for (a, b), c in top[:20]))
    print(f"首字母属于同一口型类的替换：{same}/{tot} = {100 * same / max(tot, 1):.1f}%（粗略近似）")

    # 5. 空输出
    print("\n===== 5. 空输出 =====")
    if empties.empty:
        print("无")
    else:
        e = empties.groupby("id").agg(empty=("utt_id", "size"), mean_ref_len=("ref_len", "mean")).round(1)
        e.to_csv(f"{OUT}/error_empty_{ds}.csv")
        print(e.to_string())
        print(f"所有句子的平均长度: {by_len.groupby('id').n.mean().mean():.1f} 词")

    # 6. 数字
    N = nums.groupby(["id", "has_number"]).agg(errors=("errors", "sum"), n=("n", "sum"), clips=("n", "size"))
    N["WER"] = 100 * N.errors / N.n
    pvn = N["WER"].unstack().rename(columns={False: "no_number", True: "with_number"}).round(1)
    pvn["clips_with_number"] = N["clips"].unstack().get(True)
    pvn.to_csv(f"{OUT}/error_numbers_{ds}.csv")
    print("\n===== 6. 含数字的句子 vs 不含数字 =====")
    print(pvn.reindex([m for m in REPS if m in pvn.index]).to_string())

# 错误类型图（LRS3）
t = pd.read_csv(f"{OUT}/error_types_lrs3.csv") if os.path.exists(f"{OUT}/error_types_lrs3.csv") else None
if t is not None:
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(t))
    b = np.zeros(len(t))
    for col, c in [("substitution", "#2F5D8A"), ("deletion", "#B3263E"), ("insertion", "#E0A458")]:
        ax.bar(x, t[col], bottom=b, color=c, label=col)
        b += t[col].values
    ax.set_xticks(x)
    ax.set_xticklabels([NAME[i] for i in t.id], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("WER on LRS3 (%)")
    ax.set_title("Error types: substitutions dominate")
    ax.legend(frameon=False, fontsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig9_error_types.png", dpi=300)
    plt.close(fig)
print(f"\n全部输出在 {OUT}")
