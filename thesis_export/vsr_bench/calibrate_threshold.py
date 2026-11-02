# -*- coding: utf-8 -*-
"""
calibrate_threshold.py  -  recalibrate confidence thresholds with a few clips from the new speaker

Instead of fine-tuning the model (which causes catastrophic forgetting),
this approach keeps the model unchanged and only adjusts the thresholds
for accept / correct / reject.

Experiment:
  1. Run T1 on N calibration clips from Tom Scott (from the TRAIN set)
  2. Pick thresholds on those N clips (same method as conf_tiers.py)
  3. Test on:
     - Tom Scott test set (435 clips, 10 held-out videos)
     - BBC test set (91 clips, separate video)
     - LRS3 (300 clips, other speakers - should NOT change)

Outputs:
  figs/calibrate_results.csv   : WER table for each N
  figs/calibrate_curve.png     : figure for PPT

Run in /root/autodl-tmp/vsr_bench:
  python calibrate_threshold.py
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import CACHE
from conf_utils import wer_counts

ap = argparse.ArgumentParser()
ap.add_argument("--conf-dir", default="/root/autodl-tmp/day1/conf")
ap.add_argument("--figs-dir", default="/root/autodl-tmp/day1/figs")
ap.add_argument("--accept-pct", type=float, default=0.60)
ap.add_argument("--reject-pct", type=float, default=0.15)
ap.add_argument("--model-id", default="T1", help="which model's confidence scores to use (T1 or B2)")
ap.add_argument("--n-list", nargs="+", type=int, default=[0, 20, 50, 100, 200, 500])
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
os.makedirs(a.figs_dir, exist_ok=True)

# ---- load test sets (already evaluated by eval_autoavsr_nbest.py)
def load_conf(name):
    p = os.path.join(a.conf_dir, f"{a.model_id}_{name}_confA.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, keep_default_na=False)
    df["err"], df["rlen"] = zip(*[wer_counts(r, h) for r, h in zip(df["ref"], df["hyp"])])
    return df

ts_test = load_conf("tomscott_test")
bbc = load_conf("bbc")
lrs3 = load_conf("lrs3_300")
if ts_test is None:
    raise SystemExit("run eval_autoavsr_nbest.py --dataset tomscott_test first")

# ---- load calibration pool (T1 on the TRAIN clips)
cal_path = os.path.join(a.conf_dir, f"{a.model_id}_tomscott_confA.csv")
if not os.path.exists(cal_path):
    # need to run T1 on the full tomscott set
    print(f"need calibration data: {cal_path}")
    print("run:  python -u eval_autoavsr_nbest.py --ckpt <T1> --dataset tomscott")
    raise SystemExit

cal_full = pd.read_csv(cal_path, keep_default_na=False)
# keep only TRAIN clips
split = json.load(open(os.path.join(CACHE, "tomscott_split.json")))
man = pd.read_csv(os.path.join(CACHE, "tomscott", "manifest.csv"), keep_default_na=False)
train_ids = set(man[man["video_id"].isin(split["train"])]["utt_id"])
cal_pool = cal_full[cal_full["id"].isin(train_ids)].copy()
print(f"calibration pool: {len(cal_pool)} clips from {len(split['train'])} train videos")

# ---- calibration from LRS3 (the original, cross-speaker thresholds)
lrs3_conf = load_conf("lrs3_300")
if lrs3_conf is None:
    raise SystemExit(f"run eval on lrs3_300 with {a.model_id} first")
lrs3_hi = np.quantile(lrs3_conf["confA_logp"], 1 - a.accept_pct)
lrs3_lo = np.quantile(lrs3_conf["confA_logp"], a.reject_pct)


def tiers(df, hi, lo):
    s = df["confA_logp"].clip(lower=-5)
    return np.where(s >= hi, "high", np.where(s < lo, "low", "medium"))


def metrics(df, hi, lo):
    df = df.copy()
    df["tier"] = tiers(df, hi, lo)
    all_wer = df["err"].sum() / max(df["rlen"].sum(), 1)
    kept = df[df["tier"] != "low"]
    robot_wer = kept["err"].sum() / max(kept["rlen"].sum(), 1) if len(kept) else np.nan
    hi_wer = df.loc[df["tier"] == "high", "err"].sum() / max(df.loc[df["tier"] == "high", "rlen"].sum(), 1)
    reject_pct = (df["tier"] == "low").mean()
    return {"WER_all": round(100 * all_wer, 2), "WER_robot": round(100 * robot_wer, 2),
            "WER_high": round(100 * hi_wer, 2), "reject_%": round(100 * reject_pct, 1)}


rows = []
rng = np.random.default_rng(a.seed)
for N in a.n_list:
    if N == 0:
        # use LRS3 thresholds (the "before calibration" baseline)
        hi, lo, source = lrs3_hi, lrs3_lo, "LRS3 (cross-speaker)"
    else:
        if N > len(cal_pool):
            print(f"skip N={N}: not enough calibration clips ({len(cal_pool)})")
            continue
        cal = cal_pool.sample(N, random_state=a.seed)
        cal_logp = cal["confA_logp"].clip(lower=-5)
        hi = np.quantile(cal_logp, 1 - a.accept_pct)
        lo = np.quantile(cal_logp, a.reject_pct)
        source = f"{N} Tom Scott clips"
    r = {"calibration": source, "N": N, "threshold_hi": round(hi, 3), "threshold_lo": round(lo, 3)}
    for name, df in [("tomscott_test", ts_test), ("bbc", bbc), ("lrs3", lrs3)]:
        if df is not None:
            m = metrics(df, hi, lo)
            for k, v in m.items():
                r[f"{name}_{k}"] = v
    rows.append(r)
    print(f"N={N:4d}  hi={hi:.3f} lo={lo:.3f}  "
          f"TS_robot={r.get('tomscott_test_WER_robot', '?')}  "
          f"BBC_robot={r.get('bbc_WER_robot', '?')}  "
          f"TS_reject={r.get('tomscott_test_reject_%', '?')}%")

res = pd.DataFrame(rows)
res.to_csv(os.path.join(a.figs_dir, f"calibrate_results_{a.model_id}.csv"), index=False)
pd.set_option("display.width", 200)
print("\n" + res.to_string(index=False))

# ---- figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
x = [str(n) if n > 0 else "LRS3\n(before)" for n in res["N"]]

ax = axes[0]
for col, lab, c in [("tomscott_test_WER_robot", "Tom Scott test (robot WER)", "#1f77b4"),
                     ("bbc_WER_robot", "BBC test (robot WER)", "#ff7f0e")]:
    if col in res:
        ax.plot(x, res[col], "o-", color=c, lw=2, label=lab)
ax.set_xlabel("Calibration clips from new speaker")
ax.set_ylabel("WER of accepted sentences (%)")
ax.set_title("Threshold recalibration")
ax.grid(alpha=0.3)
ax.legend()

ax = axes[1]
for col, lab, c in [("tomscott_test_reject_%", "Tom Scott test", "#1f77b4"),
                     ("bbc_reject_%", "BBC test", "#ff7f0e")]:
    if col in res:
        ax.plot(x, res[col], "s-", color=c, lw=2, label=lab)
ax.set_xlabel("Calibration clips from new speaker")
ax.set_ylabel("Rejected sentences (%)")
ax.set_title("Rejection rate after recalibration")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(os.path.join(a.figs_dir, f"calibrate_curve_{a.model_id}.png"), dpi=220)
print(f"\nsaved {a.figs_dir}/calibrate_results_{a.model_id}.csv and calibrate_curve_{a.model_id}.png")
