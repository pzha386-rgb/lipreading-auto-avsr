"""
conf_tiers.py  -  Day 2: thresholds + final figure + confidence intervals

What it does:
1. Builds a combined score  combo = z(logp) + z(confB)   (z uses LRS3 stats only)
2. Picks thresholds on LRS3 (calibration set), then applies them to BBC unchanged
   -> high (accept) / medium (LLM correction) / low (ask to repeat)
3. Bootstrap 95% CI on BBC (small set, 91 clips)
4. Two-panel risk-coverage figure for slides / thesis
5. Saves a tier CSV per set (used by Day 3 LLM correction)

Run (in /root/autodl-tmp/day1):
python -u conf_tiers.py
"""
import argparse
import math
import os
import numpy as np
import pandas as pd
from conf_utils import wer_counts

ap = argparse.ArgumentParser()
ap.add_argument("--calib", nargs="+", default=["conf/lrs3_confB.csv", "conf/T1_lrs3_300_confA.csv"])
ap.add_argument("--test", nargs="+", default=["conf/bbc_confB.csv", "conf/T1_bbc_confA.csv"])
ap.add_argument("--main", default="confA_logp", help="score used for the robot tiers")
ap.add_argument("--accept", type=float, default=0.60, help="share of LRS3 accepted directly")
ap.add_argument("--reject", type=float, default=0.15, help="share of LRS3 rejected (ask to repeat)")
ap.add_argument("--boot", type=int, default=1000)
ap.add_argument("--outdir", default="figs")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)


def load(files):
    df = pd.read_csv(files[0], keep_default_na=False)
    for f in files[1:]:
        o = pd.read_csv(f, keep_default_na=False)
        keep = ["id"] + [c for c in o.columns if c.startswith("conf") and c not in df.columns]
        df = df.merge(o[keep], on="id", how="inner")
    e, n = zip(*[wer_counts(r, h) for r, h in zip(df["ref"], df["hyp"])])
    df["err"], df["rlen"] = np.array(e), np.array(n)
    df["swer"] = df["err"] / df["rlen"].clip(lower=1)
    df["confA_logp"] = df["confA_logp"].clip(lower=-5.0)   # empty outputs (-99) -> -5
    return df


cal, tst = load(args.calib), load(args.test)

# ---- 1. combined score, normalized with LRS3 statistics only
stats = {c: (cal[c].mean(), cal[c].std() + 1e-9) for c in ["confA_logp", "confB"]}
for d in (cal, tst):
    d["combo"] = sum((d[c] - m) / s for c, (m, s) in stats.items())

SCORES = ["confA_logp", "confB", "combo"]
LABEL = {"confA_logp": "Beam score (1 model)", "confB": "Two-model agreement",
         "combo": "Combined", "random": "Random", "oracle": "Oracle"}


def risk_curve(d, order):
    e = d["err"].values[order].cumsum()
    l = d["rlen"].values[order].cumsum()
    return e / np.maximum(l, 1)


def curves(d):
    N, out = len(d), {}
    rng = np.random.default_rng(0)
    out["random"] = np.mean([risk_curve(d, rng.permutation(N)) for _ in range(200)], axis=0)
    out["oracle"] = risk_curve(d, np.argsort(d["swer"].values, kind="stable"))
    for c in SCORES:
        out[c] = risk_curve(d, np.argsort(-d[c].values, kind="stable"))
    return out


def wer(d):
    return d["err"].sum() / max(d["rlen"].sum(), 1)


# ---- 2. thresholds from LRS3, applied to BBC unchanged
hi = np.quantile(cal[args.main], 1 - args.accept)
lo = np.quantile(cal[args.main], args.reject)
print(f"thresholds on {args.main} (from LRS3):  high >= {hi:.3f}   low < {lo:.3f}\n")


def tiers(d):
    s = d[args.main]
    return np.where(s >= hi, "high", np.where(s < lo, "low", "medium"))


rows = []
for name, d in [("LRS3 (calibration)", cal), ("BBC (new speaker)", tst)]:
    d["tier"] = tiers(d)
    r = {"set": name, "N": len(d), "WER all": wer(d)}
    for t in ["high", "medium", "low"]:
        sub = d[d["tier"] == t]
        r[f"{t} %"] = len(sub) / len(d)
        r[f"{t} WER"] = wer(sub) if len(sub) else np.nan
    rows.append(r)
tier_tab = pd.DataFrame(rows)
pd.set_option("display.width", 160)
print("ROBOT TIERS  (high = accept, medium = LLM correction, low = ask to repeat)")
print(tier_tab.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
tier_tab.to_csv(f"{args.outdir}/tiers_summary.csv", index=False)

# ---- 3. summary + bootstrap CI on BBC
def summary(d, c_dict):
    N = len(d)
    k = lambda cov: max(math.ceil(cov * N) - 1, 0)
    return {c: {"AURC": v.mean(), "WER@60%": v[k(0.6)], "WER@80%": v[k(0.8)]} for c, v in c_dict.items()}


cal_c, tst_c = curves(cal), curves(tst)
rng = np.random.default_rng(1)
boot = {c: [] for c in SCORES + ["random"]}
for _ in range(args.boot):
    b = tst.iloc[rng.integers(0, len(tst), len(tst))].reset_index(drop=True)
    s = summary(b, curves(b))
    for c in boot:
        boot[c].append((s[c]["AURC"], s[c]["WER@60%"]))

out = []
for name, d, cd in [("LRS3", cal, cal_c), ("BBC", tst, tst_c)]:
    s = summary(d, cd)
    for c in SCORES + ["random", "oracle"]:
        row = {"set": name, "score": LABEL[c], **s[c]}
        if name == "BBC" and c in boot:
            arr = np.array(boot[c])
            row["AURC 95% CI"] = f"[{np.percentile(arr[:, 0], 2.5):.3f}, {np.percentile(arr[:, 0], 97.5):.3f}]"
            row["WER@60% 95% CI"] = f"[{np.percentile(arr[:, 1], 2.5):.3f}, {np.percentile(arr[:, 1], 97.5):.3f}]"
        out.append(row)
res = pd.DataFrame(out).fillna("")
print("\nRISK-COVERAGE SUMMARY")
print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
res.to_csv(f"{args.outdir}/conf_final_summary.csv", index=False)

# paired test: is the main score better than random on BBC?
arr_m, arr_r = np.array(boot[args.main])[:, 0], np.array(boot["random"])[:, 0]
p = (arr_m >= arr_r).mean()
print(f"\nBBC bootstrap: {LABEL[args.main]} AURC lower than random in {100 * (1 - p):.1f}% of samples (p = {p:.3f})")

# ---- 4. figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
style = {"random": dict(ls="--", c="gray"), "oracle": dict(ls=":", c="black"),
         "confA_logp": dict(c="#1f77b4", lw=2.2), "confB": dict(c="#ff7f0e", lw=1.8),
         "combo": dict(c="#2ca02c", lw=1.8)}
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (title, d, cd) in zip(axes, [("LRS3 (seen speakers)", cal, cal_c), ("BBC (new speaker)", tst, tst_c)]):
    cov = np.arange(1, len(d) + 1) / len(d) * 100
    for c in ["random", "oracle", "confB", "combo", "confA_logp"]:
        ax.plot(cov, cd[c] * 100, label=LABEL[c], **style[c])
    ax.axvline(args.accept * 100, color="red", alpha=0.3, lw=1)
    ax.set_title(title)
    ax.set_xlabel("Accepted sentences (%)")
    ax.set_xlim(5, 100)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("WER of accepted sentences (%)")
axes[1].legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(f"{args.outdir}/fig_risk_coverage_final.png", dpi=250)
print(f"\nsaved {args.outdir}/fig_risk_coverage_final.png")

# ---- 5. tier CSVs for Day 3 (LLM correction)
for name, d in [("lrs3", cal), ("bbc", tst)]:
    p_ = f"conf/{name}_tiers.csv"
    d[["id", "ref", "hyp", "confA_logp", "confB", "combo", "tier", "err", "rlen"]].to_csv(p_, index=False)
    print(f"saved {p_}")
