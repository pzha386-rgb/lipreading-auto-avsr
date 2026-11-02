"""
agreement.py  -  Method B: two-model agreement (no new inference needed)

Input : two prediction CSVs you already have (e.g. Auto-AVSR and BRAVEn),
        each with columns: id, ref, hyp   (names can be changed by args)
Output: one CSV with id, ref, hyp (from model A), confB

Example:
python -u agreement.py --a preds/autoavsr_bbc.csv --b preds/braven_bbc.csv \
       --out conf/bbc_confB.csv
"""
import argparse
import os
import pandas as pd
from conf_utils import text_disagreement

ap = argparse.ArgumentParser()
ap.add_argument("--a", required=True, help="main model predictions (its hyp is kept)")
ap.add_argument("--b", required=True, help="second model predictions")
ap.add_argument("--out", required=True)
ap.add_argument("--id-col", default="id")
ap.add_argument("--ref-col", default="ref")
ap.add_argument("--hyp-col", default="hyp")
args = ap.parse_args()

A = pd.read_csv(args.a, keep_default_na=False)
B = pd.read_csv(args.b, keep_default_na=False)
# keep only the needed columns FIRST, then rename (avoids duplicate names)
A = A[[args.id_col, args.ref_col, args.hyp_col]].copy()
A.columns = ["id", "ref", "hyp"]
B = B[[args.id_col, args.hyp_col]].copy()
B.columns = ["id", "hyp_b"]

m = A.merge(B, on="id", how="inner")
print(f"A={len(A)}  B={len(B)}  matched={len(m)}")
if len(m) < min(len(A), len(B)):
    print("WARNING: some ids did not match - check the id column in both files")

m["confB"] = [1.0 - text_disagreement(a, b) for a, b in zip(m["hyp"], m["hyp_b"])]

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
m.to_csv(args.out, index=False)
print(f"saved {args.out}")
print(f"fully agree: {(m['confB'] == 1).mean():.1%}   mean confB: {m['confB'].mean():.3f}")
