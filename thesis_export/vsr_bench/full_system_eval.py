# -*- coding: utf-8 -*-
"""
full_system_eval.py  -  evaluate the COMPLETE reliable lip-reading module

Combines all methods into one pipeline and shows the cumulative improvement:
  Step 0: raw model output (baseline)
  Step 1: + confidence-based rejection (remove low-confidence sentences)
  Step 2: + threshold recalibration (use Tom Scott calibration data)
  Step 3: + LLM correction on medium-confidence sentences
  Step 4: LoRA-adapted model (replaces step 0 model)
  Step 5: LoRA + calibration + rejection
  Step 6: LoRA + calibration + rejection + LLM correction (full system)

Also compares: which method contributes the most?

Run in /root/autodl-tmp/vsr_bench:
  python full_system_eval.py
"""
import json
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/root/autodl-tmp/day1")
from conf_utils import wer_counts, normalize

CONF = "/root/autodl-tmp/day1/conf"
CACHE = "/root/autodl-tmp/cache"
FIGS = "/root/autodl-tmp/day1/figs"
os.makedirs(FIGS, exist_ok=True)


def load(model, ds):
    p = f"{CONF}/{model}_{ds}_confA.csv"
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p, keep_default_na=False)
    e, n = zip(*[wer_counts(r, h) for r, h in zip(df["ref"], df["hyp"])])
    df["err"], df["rlen"] = np.array(e), np.array(n)
    return df


def load_llm(ds, model_tag="k5_choose"):
    p = f"{CONF}/{ds}_llm_{model_tag}.csv"
    if not os.path.exists(p):
        return None
    return pd.read_csv(p, keep_default_na=False)


def wer(df, err_col="err"):
    return 100 * df[err_col].sum() / max(df["rlen"].sum(), 1)


def robot_wer(df, err_col="err"):
    kept = df[df["tier"] != "low"]
    return 100 * kept[err_col].sum() / max(kept["rlen"].sum(), 1) if len(kept) else np.nan


def apply_tiers(df, hi, lo):
    s = df["confA_logp"].clip(lower=-5)
    df["tier"] = np.where(s >= hi, "high", np.where(s < lo, "low", "medium"))
    return df


def apply_llm(df, llm_df):
    """Replace hyp with LLM output for medium-confidence sentences"""
    if llm_df is None:
        return df
    llm_map = dict(zip(llm_df["id"], llm_df["llm"]))
    new_err = []
    for _, r in df.iterrows():
        if r["tier"] == "medium" and r["id"] in llm_map:
            e, _ = wer_counts(r["ref"], llm_map[r["id"]])
            new_err.append(e)
        else:
            new_err.append(r["err"])
    df["err_llm"] = new_err
    return df


# ---- Load data
split = json.load(open(os.path.join(CACHE, "tomscott_split.json")))
man = pd.read_csv(os.path.join(CACHE, "tomscott", "manifest.csv"), keep_default_na=False)
train_ids = set(man[man["video_id"].isin(split["train"])]["utt_id"])

SETS = {
    "tomscott_test": "Tom Scott test (435 clips)",
    "bbc": "BBC test (91 clips)",
}
MODELS = {
    "T1": {"label": "Auto-AVSR (T1)", "lora": None},
    "L15": {"label": "T1 + LoRA 15min", "lora": "L15"},
}

# Check which LoRA models exist
for lid in ["L5", "L15", "L30", "L60", "L120", "Lall"]:
    p = f"{CONF}/{lid}_tomscott_test_confA.csv"
    if os.path.exists(p):
        MODELS[lid] = {"label": f"T1 + LoRA ({lid[1:]} min)", "lora": lid}

# ---- LRS3 thresholds (cross-speaker, no calibration)
lrs3 = load("T1", "lrs3_300")
lrs3_hi = np.quantile(lrs3["confA_logp"], 0.40)  # top 60%
lrs3_lo = np.quantile(lrs3["confA_logp"], 0.15)

# ---- Tom Scott calibration thresholds (20 clips)
t1_full = load("T1", "tomscott")
if t1_full is not None:
    cal = t1_full[t1_full["id"].isin(train_ids)].sample(20, random_state=0)
    cal_hi = np.quantile(cal["confA_logp"].clip(lower=-5), 0.40)
    cal_lo = np.quantile(cal["confA_logp"].clip(lower=-5), 0.15)
else:
    cal_hi, cal_lo = lrs3_hi, lrs3_lo

# ---- LLM correction data
llm_bbc = load_llm("bbc")
llm_lrs3 = load_llm("lrs3")

print("=" * 80)
print("COMPLETE SYSTEM EVALUATION")
print("=" * 80)

all_rows = []
for ds, ds_label in SETS.items():
    print(f"\n{'='*60}")
    print(f"  {ds_label}")
    print(f"{'='*60}")

    for mid, minfo in MODELS.items():
        df = load(mid, ds)
        if df is None:
            continue

        # Step 0: raw
        raw = wer(df)

        # Step 1: + rejection (LRS3 thresholds)
        df1 = apply_tiers(df.copy(), lrs3_hi, lrs3_lo)
        rej_lrs3 = robot_wer(df1)
        rej_pct_lrs3 = 100 * (df1["tier"] == "low").mean()

        # Step 2: + calibrated rejection (Tom Scott thresholds)
        # For LoRA models, use their own confidence for calibration
        if mid != "T1":
            lora_full = load(mid, "tomscott")
            if lora_full is not None:
                lora_cal = lora_full[lora_full["id"].isin(train_ids)].sample(min(20, len(lora_full)), random_state=0)
                m_hi = np.quantile(lora_cal["confA_logp"].clip(lower=-5), 0.40)
                m_lo = np.quantile(lora_cal["confA_logp"].clip(lower=-5), 0.15)
            else:
                m_hi, m_lo = cal_hi, cal_lo
        else:
            m_hi, m_lo = cal_hi, cal_lo

        df2 = apply_tiers(df.copy(), m_hi, m_lo)
        rej_cal = robot_wer(df2)
        rej_pct_cal = 100 * (df2["tier"] == "low").mean()

        # Step 3: + LLM correction
        llm = llm_bbc if ds == "bbc" else llm_lrs3
        df3 = apply_llm(df2.copy(), llm)
        if "err_llm" in df3.columns:
            rej_llm = robot_wer(df3, "err_llm")
        else:
            rej_llm = rej_cal

        row = {
            "dataset": ds_label,
            "model": minfo["label"],
            "WER_raw": round(raw, 2),
            "WER_reject_LRS3": round(rej_lrs3, 2),
            f"reject_%_LRS3": round(rej_pct_lrs3, 1),
            "WER_reject_calibrated": round(rej_cal, 2),
            f"reject_%_calibrated": round(rej_pct_cal, 1),
            "WER_full_system": round(rej_llm, 2),
        }
        all_rows.append(row)
        print(f"\n  {minfo['label']}:")
        print(f"    Raw WER:                    {raw:.2f}%")
        print(f"    + rejection (LRS3):         {rej_lrs3:.2f}%  (reject {rej_pct_lrs3:.0f}%)")
        print(f"    + calibrated rejection:     {rej_cal:.2f}%  (reject {rej_pct_cal:.0f}%)")
        print(f"    + LLM correction (full):    {rej_llm:.2f}%")

res = pd.DataFrame(all_rows)
res.to_csv(f"{FIGS}/full_system_results.csv", index=False)

# ---- Summary: best system on each test set
print(f"\n{'='*60}")
print("BEST SYSTEM per test set:")
for ds, ds_label in SETS.items():
    sub = res[res["dataset"] == ds_label]
    if len(sub):
        best = sub.loc[sub["WER_full_system"].idxmin()]
        print(f"  {ds_label}: {best['WER_full_system']:.2f}% ({best['model']})")
        baseline = sub.loc[sub["WER_raw"].idxmax()]
        print(f"    improvement: {baseline['WER_raw']:.2f}% -> {best['WER_full_system']:.2f}% "
              f"(↓{baseline['WER_raw'] - best['WER_full_system']:.2f} percentage points)")

# ---- Waterfall chart data (for PPT)
print(f"\n{'='*60}")
print("WATERFALL (Tom Scott test, using best LoRA):")
ts = res[res["dataset"].str.contains("Tom Scott")]
if len(ts):
    t1 = ts[ts["model"].str.contains("T1")].iloc[0] if len(ts[ts["model"].str.contains("T1")]) else None
    best_lora = ts.loc[ts["WER_full_system"].idxmin()]
    if t1 is not None:
        steps = [
            ("Original T1", t1["WER_raw"]),
            ("+ LoRA adaptation", best_lora["WER_raw"]),
            ("+ confidence rejection", best_lora["WER_reject_LRS3"]),
            ("+ threshold calibration", best_lora["WER_reject_calibrated"]),
            ("+ LLM correction", best_lora["WER_full_system"]),
        ]
        for name, val in steps:
            print(f"  {name:30s}  {val:.2f}%")

print(f"\nsaved {FIGS}/full_system_results.csv")
