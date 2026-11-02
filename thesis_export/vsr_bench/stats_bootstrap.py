# -*- coding: utf-8 -*-
"""
统计显著性：配对 bootstrap（10,000 次有放回抽样，按片段重采样）
  - 每个模型在每个数据集上的 WER 95% 置信区间
  - 关键对比的 WER 差值、95% 置信区间、p 值
差值 = B - A（负数表示 B 更好）；置信区间不含 0 即为显著

输出：/root/autodl-tmp/figures/stats_ci.csv, stats_pairs.csv, stats_robustness.csv
运行：python stats_bootstrap.py
"""
import os

import numpy as np
import pandas as pd

from common import ROOT

R = f"{ROOT}/results_v2"
OUT = f"{ROOT}/figures"
os.makedirs(OUT, exist_ok=True)
N_BOOT, SEED = 10000, 2026


def load(mid, ds):
    p = f"{R}/{mid}_{ds}.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, dtype={"utt_id": str})
    return df.set_index("utt_id")[["errors", "n_words"]]


def boot_idx(n):
    return np.random.default_rng(SEED).integers(0, n, size=(N_BOOT, n))


def ci_single(df):
    e, n = df.errors.values, df.n_words.values
    idx = boot_idx(len(df))
    w = 100 * e[idx].sum(1) / n[idx].sum(1)
    return 100 * e.sum() / n.sum(), np.percentile(w, 2.5), np.percentile(w, 97.5)


def paired(a, b):
    """返回 (WER_A, WER_B, 差值 B-A, 下限, 上限, p 值)"""
    j = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    ea, eb, n = j.errors_a.values, j.errors_b.values, j.n_words_a.values
    idx = boot_idx(len(j))
    d = 100 * (eb[idx].sum(1) - ea[idx].sum(1)) / n[idx].sum(1)
    obs = 100 * (eb.sum() - ea.sum()) / n.sum()
    p = 2 * min((d >= 0).mean(), (d <= 0).mean())          # 双侧
    return 100 * ea.sum() / n.sum(), 100 * eb.sum() / n.sum(), obs, \
        np.percentile(d, 2.5), np.percentile(d, 97.5), min(p, 1.0)


# ---------------- 1. 每个模型的置信区间 ----------------
rows = []
for f in sorted(os.listdir(R)):
    if not f.endswith(".csv") or "_limit" in f or f.startswith("summary") or "_bbc_speed" in f:
        continue
    mid, ds = f[:-4].split("_", 1)
    if ds not in ("bbc", "lrs3_300", "lrs3"):
        continue
    df = load(mid, ds)
    w, lo, hi = ci_single(df)
    rows.append({"id": mid, "dataset": ds, "n_clips": len(df), "wer": round(w, 2),
                 "ci_low": round(lo, 2), "ci_high": round(hi, 2), "ci_width": round(hi - lo, 2)})
ci = pd.DataFrame(rows)
ci.to_csv(f"{OUT}/stats_ci.csv", index=False)
print("===== 1. 置信区间的平均宽度（越窄越可靠）=====")
print(ci.groupby("dataset").agg(clips=("n_clips", "first"), mean_ci_width=("ci_width", "mean")).round(2))

# ---------------- 2. 关键对比 ----------------
PAIRS = [
    ("监督学习：标签 438h → 1759h", "S0", "S1"),
    ("监督学习：标签 1759h → 3291h", "S1", "T1"),
    ("自监督 30h vs 监督 3291h", "T1", "T3"),
    ("Self-training：BRAVEn 低资源", "B1", "T3"),
    ("Self-training：BRAVEn 高资源", "B3", "B4"),
    ("标签 30h → 433h：BRAVEn", "T3", "B4"),
    ("标签 30h → 433h：RAVEn", "T4", "R2"),
    ("标签 30h → 433h：USR2 Large", "U1", "U3"),
    ("LM：BRAVEn 低资源", "T3", "B2"),
    ("LM：BRAVEn 高资源", "B4", "B5"),
    ("LM：RAVEn 低资源", "T4", "R1"),
    ("LM：RAVEn 高资源", "R2", "R3"),
    ("LM：Auto-AVSR (VSR-ML)", "S2", "S2lm"),
    ("LM：Ma2022", "S4", "S4lm"),
    ("模型大小：USR2 Base+ → Large", "U2", "U3"),
    ("模型大小：USR2 Large → Huge", "U3", "U4"),
    ("解码：USR2 Huge beam40 → greedy", "U4", "U4g"),
    ("解码：USR2 Huge beam40 → beam10", "U4", "U4b10"),
    ("BRAVEn vs RAVEn（同为低资源 +ST）", "T4", "T3"),
    ("排名翻转：T1 vs U4", "T1", "U4"),
    ("T1 vs S2（是否同一模型）", "T1", "S2"),
]
rows = []
for ds in ["lrs3", "lrs3_300", "bbc"]:
    for label, a, b in PAIRS:
        da, db = load(a, ds), load(b, ds)
        if da is None or db is None:
            continue
        wa, wb, d, lo, hi, p = paired(da, db)
        rows.append({"dataset": ds, "comparison": label, "A": a, "B": b, "wer_A": round(wa, 2),
                     "wer_B": round(wb, 2), "diff_B_minus_A": round(d, 2), "ci_low": round(lo, 2),
                     "ci_high": round(hi, 2), "p": round(p, 4),
                     "significant": "yes" if (lo > 0 or hi < 0) else "no"})
pairs = pd.DataFrame(rows)
pairs.to_csv(f"{OUT}/stats_pairs.csv", index=False)

pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 40)
for ds in ["lrs3", "bbc"]:
    print(f"\n===== 2. 关键对比：{ds}（差值 = B − A，负数 = B 更好）=====")
    t = pairs[pairs.dataset == ds][["comparison", "A", "B", "wer_A", "wer_B", "diff_B_minus_A",
                                    "ci_low", "ci_high", "significant"]]
    print(t.to_string(index=False))

# ---------------- 3. 鲁棒性：每种扰动相对干净 BBC 的变化 ----------------
CONDS = ["bbc_blur2", "bbc_blur4", "bbc_low50", "bbc_low25", "bbc_dark05", "bbc_dark03", "bbc_drop25", "bbc_drop50"]
rows = []
for mid in ["T1", "S4", "T3", "T4", "B4", "U4"]:
    clean = load(mid, "bbc")
    for c in CONDS:
        pert = load(mid, c)
        if clean is None or pert is None:
            continue
        wa, wb, d, lo, hi, p = paired(clean, pert)
        rows.append({"id": mid, "condition": c.replace("bbc_", ""), "clean": round(wa, 2),
                     "perturbed": round(wb, 2), "increase": round(d, 2), "ci_low": round(lo, 2),
                     "ci_high": round(hi, 2), "significant": "yes" if (lo > 0 or hi < 0) else "no"})
rob = pd.DataFrame(rows)
rob.to_csv(f"{OUT}/stats_robustness.csv", index=False)
print("\n===== 3. 鲁棒性：哪些扰动带来显著变化（yes = 显著）=====")
print(rob.pivot(index="id", columns="condition", values="significant").to_string())

print(f"\n全部结果保存在 {OUT}/stats_*.csv")
