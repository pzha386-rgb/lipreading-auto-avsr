# -*- coding: utf-8 -*-
"""
reliability_diagram.py  -  confidence calibration plot

For each confidence bin, plot: predicted confidence vs actual accuracy.
A perfectly calibrated model lies on the diagonal.
Shows that T1 is overconfident on new speakers; B2 is better calibrated.

Run in /root/autodl-tmp/vsr_bench:
  python reliability_diagram.py
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/day1")
from conf_utils import wer_counts

CONF = "/root/autodl-tmp/day1/conf"
FIGS = "/root/autodl-tmp/day1/figs"
os.makedirs(FIGS, exist_ok=True)


def load(model, ds):
    p = f"{CONF}/{model}_{ds}_confA.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, keep_default_na=False)
    e, n = zip(*[wer_counts(r, h) for r, h in zip(df["ref"], df["hyp"])])
    df["err"], df["rlen"] = np.array(e), np.array(n)
    df["acc"] = 1.0 - df["err"] / df["rlen"].clip(lower=1)
    df["conf"] = df["confA_logp"].clip(lower=-5)
    # normalize to 0-1 range per model+dataset for fair comparison
    df["conf_norm"] = (df["conf"] - df["conf"].min()) / (df["conf"].max() - df["conf"].min() + 1e-9)
    return df


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
bins = np.linspace(0, 1, 11)
bin_centers = (bins[:-1] + bins[1:]) / 2

for ax, (ds, title) in zip(axes, [("lrs3_300", "LRS3 (seen speakers)"),
                                    ("tomscott_test", "Tom Scott test (new speaker)")]):
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect calibration")

    for model, color, marker, label in [("T1", "#1f77b4", "o", "Auto-AVSR (T1)"),
                                         ("B2", "#ff7f0e", "s", "BRAVEn (B2)")]:
        df = load(model, ds)
        if df is None:
            continue
        accs, confs, counts = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (df["conf_norm"] >= lo) & (df["conf_norm"] < hi)
            if mask.sum() >= 3:
                accs.append(df.loc[mask, "acc"].mean())
                confs.append(df.loc[mask, "conf_norm"].mean())
                counts.append(mask.sum())
        ax.plot(confs, accs, f"-{marker}", color=color, lw=2, markersize=6, label=label)

    ax.set_xlabel("Normalized confidence")
    ax.set_ylabel("Actual accuracy (1 - sentence WER)")
    ax.set_title(title)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")

plt.tight_layout()
plt.savefig(f"{FIGS}/reliability_diagram.png", dpi=220)
print(f"saved {FIGS}/reliability_diagram.png")

# ---- Expected Calibration Error (ECE)
print("\nExpected Calibration Error (lower = better calibrated):")
for model in ["T1", "B2"]:
    for ds, label in [("lrs3_300", "LRS3"), ("tomscott_test", "Tom Scott"), ("bbc", "BBC")]:
        df = load(model, ds)
        if df is None:
            continue
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (df["conf_norm"] >= lo) & (df["conf_norm"] < hi)
            if mask.sum() >= 1:
                ece += mask.sum() / len(df) * abs(df.loc[mask, "acc"].mean() - df.loc[mask, "conf_norm"].mean())
        print(f"  {model} on {label:12s}: ECE = {ece:.3f}")
