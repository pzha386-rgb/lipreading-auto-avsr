"""
conf_analysis.py  -  Is the confidence useful?  (Day 1 check + Day 2 main figure)

Input : one or more CSVs merged on "id".
        The FIRST file must have: id, ref, hyp
        Other columns starting with "conf" are treated as confidence scores.
Output: printed table + risk-coverage plot (png) + curve data (csv)

Example:
python -u conf_analysis.py --files conf/bbc_confB.csv --name bbc --outdir figs
python -u conf_analysis.py --files conf/bbc_confB.csv conf/bbc_confA.csv --name bbc --outdir figs
"""
import argparse
import math
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from conf_utils import wer_counts

ap = argparse.ArgumentParser()
ap.add_argument("--files", nargs="+", required=True)
ap.add_argument("--name", default="test")
ap.add_argument("--outdir", default="figs")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.files[0], keep_default_na=False)
for f in args.files[1:]:
    other = pd.read_csv(f, keep_default_na=False)
    keep = ["id"] + [c for c in other.columns if c.startswith("conf") and c not in df.columns]
    df = df.merge(other[keep], on="id", how="inner")

errs, lens = zip(*[wer_counts(r, h) for r, h in zip(df["ref"], df["hyp"])])
df["err"], df["rlen"] = np.array(errs), np.array(lens)
df["swer"] = df["err"] / df["rlen"].clip(lower=1)
N = len(df)
conf_cols = [c for c in df.columns if c.startswith("conf")]
print(f"[{args.name}] sentences={N}  overall WER={df['err'].sum() / df['rlen'].sum():.2%}")
print(f"confidence columns: {conf_cols}")


def curve(order):
    e = df["err"].values[order].cumsum()
    l = df["rlen"].values[order].cumsum()
    return e / np.maximum(l, 1)          # risk when keeping top n, n=1..N


rows, curves = [], {}
rng = np.random.default_rng(0)
rand = np.mean([curve(rng.permutation(N)) for _ in range(200)], axis=0)
oracle = curve(np.argsort(df["swer"].values, kind="stable"))
curves["random"], curves["oracle"] = rand, oracle

for c in conf_cols:
    order = np.argsort(-df[c].values, kind="stable")   # most confident first
    r = curve(order)
    curves[c] = r
    rho, p = spearmanr(df[c], df["swer"])
    at = lambda cov: r[max(math.ceil(cov * N) - 1, 0)]
    rows.append({"conf": c, "spearman": rho, "p": p, "AURC": r.mean(),
                 "WER@60%": at(0.6), "WER@80%": at(0.8), "WER@100%": at(1.0)})

for base in ["random", "oracle"]:
    r = curves[base]
    rows.append({"conf": base, "spearman": np.nan, "p": np.nan, "AURC": r.mean(),
                 "WER@60%": r[math.ceil(0.6 * N) - 1], "WER@80%": r[math.ceil(0.8 * N) - 1],
                 "WER@100%": r[-1]})

res = pd.DataFrame(rows)
pd.set_option("display.width", 140)
print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
res.to_csv(f"{args.outdir}/{args.name}_conf_summary.csv", index=False)

cov = np.arange(1, N + 1) / N
pd.DataFrame({"coverage": cov, **curves}).to_csv(f"{args.outdir}/{args.name}_risk_coverage.csv", index=False)

# ---- go / no-go hint
good = [r for r in rows if r["conf"] in conf_cols and abs(r["spearman"]) > 0.3 and r["AURC"] < rand.mean()]
print("\nDAY-1 CHECK:", "GO - confidence works" if good else
      "WEAK - discuss before Day 2 (plan B: LLM correction as main line)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.figure(figsize=(6, 4.2))
plt.plot(cov * 100, rand * 100, "--", color="gray", label="Random")
plt.plot(cov * 100, oracle * 100, ":", color="black", label="Oracle")
for c in conf_cols:
    plt.plot(cov * 100, curves[c] * 100, label=c, linewidth=2)
plt.xlabel("Accepted sentences (%)")
plt.ylabel("WER of accepted sentences (%)")
plt.title(f"Risk-coverage ({args.name})")
plt.xlim(5, 100)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(f"{args.outdir}/{args.name}_risk_coverage.png", dpi=200)
print(f"saved {args.outdir}/{args.name}_risk_coverage.png")
