# -*- coding: utf-8 -*-
"""
adapt_summary.py  -  table + figure for the speaker adaptation experiment
  python adapt_summary.py
Output: /root/autodl-tmp/day1/figs/adapt_results.csv, adapt_curve.png
"""
import json
import os

import numpy as np
import pandas as pd

CONF = "/root/autodl-tmp/day1/conf"
FIGS = "/root/autodl-tmp/day1/figs"
ADAPT = "/root/autodl-tmp/adapt_v2"
os.makedirs(FIGS, exist_ok=True)

RUNS = [("T1", 0, None), ("A5", 5, "m5"), ("A15", 15, "m15"), ("A30", 30, "m30"),
        ("A60", 60, "m60"), ("A120", 120, "m120"), ("Aall", None, "m0")]
SETS = [("tomscott_test", "Tom Scott test (10 new videos)"), ("bbc", "BBC test (91 clips)"),
        ("lrs3_300", "LRS3 (forgetting check)")]


def aurc(df):
    """area under risk-coverage curve with the beam-score confidence (lower = better)"""
    d = df[df["confA_logp"] > -50].sort_values("confA_logp", ascending=False)
    return float((d["errors"].cumsum() / d["n_words"].cumsum().clip(lower=1)).mean())


rows = []
for rid, mins, folder in RUNS:
    info = {}
    if folder and os.path.exists(f"{ADAPT}/autoavsr_{folder}/history.json"):
        h = json.load(open(f"{ADAPT}/autoavsr_{folder}/history.json"))
        info = {"train_min": round(h["train_min"], 1), "epochs": len(h["history"]) - 1,
                "train_time_min": round(h["history"][-1].get("minutes", 0), 1),
                "val_loss_start": round(h["history"][0]["val_loss"], 2),
                "val_loss_best": round(min(x["val_loss"] for x in h["history"]), 2)}
    elif rid == "T1":
        info = {"train_min": 0}
    r = {"model": rid, **info}
    for ds, _ in SETS:
        p = f"{CONF}/{rid}_{ds}_confA.csv"
        if os.path.exists(p):
            df = pd.read_csv(p, keep_default_na=False)
            r[f"WER_{ds}"] = round(100 * df["errors"].sum() / df["n_words"].sum(), 2)
            r[f"AURC_{ds}"] = round(aurc(df), 3)
    rows.append(r)

res = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print(res.to_string(index=False))
res.to_csv(f"{FIGS}/adapt_results.csv", index=False)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
have = res[res.get("WER_tomscott_test").notna()] if "WER_tomscott_test" in res else res.iloc[:0]
if len(have):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = list(range(len(have)))
    labels = [("0\n(no adapt)" if m == "T1" else f"{t:g}") for m, t in zip(have["model"], have["train_min"].fillna(0))]
    ax = axes[0]
    for ds, lab in SETS[:2]:
        if f"WER_{ds}" in have:
            ax.plot(x, have[f"WER_{ds}"], "o-", lw=2, label=lab)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Minutes of Tom Scott used for adaptation")
    ax.set_ylabel("WER (%)")
    ax.set_title("New speaker: WER after adaptation")
    ax.grid(alpha=0.3)
    ax.legend()
    ax = axes[1]
    if "WER_lrs3_300" in have:
        ax.plot(x, have["WER_lrs3_300"], "s-", color="gray", lw=2, label="LRS3 (other speakers)")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Minutes of Tom Scott used for adaptation")
    ax.set_ylabel("WER (%)")
    ax.set_title("Forgetting check")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIGS}/adapt_curve.png", dpi=220)
    print(f"\nsaved {FIGS}/adapt_results.csv and adapt_curve.png")
