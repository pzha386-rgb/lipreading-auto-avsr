"""
llm_correct.py  -  LLM correction of lip-reading output (uses the N-best list)

Input : conf/<set>_tiers.csv           (from conf_tiers.py)
        conf/T1_<nbest_name>_nbest.jsonl (from eval_autoavsr_nbest.py)
Output: conf/<set>_llm_k<K>.csv        (every sentence: before / after)
        figs/llm_summary_<set>_k<K>.csv

It corrects EVERY sentence once, then compares 3 strategies:
  none        : no correction
  medium only : correct only medium-confidence sentences   (our design)
  all         : correct every sentence                     (to show the risk)

Run (in /root/autodl-tmp/day1):
  python -u llm_correct.py --set bbc  --limit 5          # quick test
  python -u llm_correct.py --set bbc
  python -u llm_correct.py --set lrs3
  python -u llm_correct.py --set bbc --k 1               # ablation: no N-best
"""
import argparse
import json
import os
import re
import time
import numpy as np
import pandas as pd
from conf_utils import wer_counts, normalize

ap = argparse.ArgumentParser()
ap.add_argument("--set", required=True, choices=["bbc", "lrs3"])
ap.add_argument("--model", default="/root/autodl-tmp/llm/Qwen2.5-7B-Instruct")
ap.add_argument("--k", type=int, default=5, help="how many N-best candidates to show the LLM")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--mode", default="rewrite", choices=["rewrite", "choose"],
                help="rewrite: LLM writes a sentence; choose: LLM only picks one candidate")
ap.add_argument("--dry-run", action="store_true", help="no LLM, just test the pipeline")
args = ap.parse_args()

NBEST_FILE = {"bbc": "conf/T1_bbc_nbest.jsonl", "lrs3": "conf/T1_lrs3_300_nbest.jsonl"}[args.set]
tiers = pd.read_csv(f"conf/{args.set}_tiers.csv", keep_default_na=False)
nbest = {}
for line in open(NBEST_FILE, encoding="utf-8"):
    d = json.loads(line)
    nbest[d["id"]] = [h["text"] for h in d["nbest"]]
tiers = tiers[tiers["id"].isin(nbest)].reset_index(drop=True)
if args.limit:
    tiers = tiers.iloc[:args.limit]
print(f"{args.set}: {len(tiers)} sentences   tiers: {tiers['tier'].value_counts().to_dict()}")

SYSTEM = (
    "You fix errors in automatic lip-reading output. "
    "Lip reading often confuses words that look the same on the lips, "
    "for example p/b/m, f/v, t/d/n, k/g, and short words like a/the/and. "
    "You get the top candidates from the lip-reading model, best first. "
    "Write the sentence the speaker most likely said. "
    "Prefer words from the candidates. Make only small changes. "
    "Do not add new information. "
    "Output only the corrected sentence in lowercase, nothing else."
)


SYSTEM_CHOOSE = (
    "You help an automatic lip-reading system. "
    "You get the top candidates for one spoken English sentence, best first. "
    "Lip reading confuses words that look the same on the lips. "
    "Pick the candidate that is the most natural and meaningful English sentence. "
    "If they are all similar, pick 1. "
    "Answer with the number only."
)


def make_prompt(cands):
    lines = "\n".join(f"{i}. {c.lower()}" for i, c in enumerate(cands, 1))
    return f"Candidates:\n{lines}\n\nCorrected sentence:"


def clean(text, fallback):
    t = text.strip().split("\n")[0]
    t = re.sub(r"^(corrected sentence:|sentence:)\s*", "", t, flags=re.I).strip(" \"'")
    nf, nt = len(normalize(fallback).split()), len(normalize(t).split())
    # safety: empty output or big length change -> keep the original
    if nt == 0 or (nf > 0 and not (0.5 <= nt / nf <= 1.6)):
        return fallback, True
    return t, False


if not args.dry_run:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    import transformers
    dkey = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    llm = AutoModelForCausalLM.from_pretrained(args.model, device_map="cuda", **{dkey: torch.bfloat16}).eval()
    print(f"loaded {args.model}")


def correct(cands):
    if args.dry_run:
        return cands[0]
    sys_msg = SYSTEM_CHOOSE if args.mode == "choose" else SYSTEM
    user = make_prompt(cands).replace("Corrected sentence:", "Best candidate number:") if args.mode == "choose" else make_prompt(cands)
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
    enc = {k: v.to("cuda") for k, v in enc.items()}
    with torch.no_grad():
        out = llm.generate(**enc, max_new_tokens=80, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


rows = []
t_all = time.time()
for i, r in tiers.iterrows():
    cands = nbest[r["id"]][:args.k]
    hyp0 = r["hyp"]
    if not cands or not normalize(cands[0]):
        new, fell_back, sec = hyp0, True, 0.0
    else:
        t0 = time.perf_counter()
        raw = correct(cands)
        sec = time.perf_counter() - t0
        if args.mode == "choose":
            m = re.search(r"\d+", raw)
            idx = int(m.group()) - 1 if m else 0
            fell_back = not (m and 0 <= idx < len(cands))
            new = cands[idx] if not fell_back else hyp0
        else:
            new, fell_back = clean(raw, hyp0)
    e0, n = wer_counts(r["ref"], hyp0)
    e1, _ = wer_counts(r["ref"], new)
    rows.append({"id": r["id"], "tier": r["tier"], "ref": r["ref"], "hyp": hyp0,
                 "llm": normalize(new), "err_before": e0, "err_after": e1, "n_words": n,
                 "fallback": fell_back, "llm_sec": round(sec, 3)})
    if args.limit or (i + 1) % 25 == 0:
        print(f"[{i + 1}/{len(tiers)}] {r['tier']:6s} errors {e0} -> {e1}  ({sec:.2f}s)")
        if args.limit:
            print(f"   REF: {r['ref']}\n   T1 : {hyp0}\n   LLM: {normalize(new)}")

df = pd.DataFrame(rows)
os.makedirs("figs", exist_ok=True)
TAG = f"k{args.k}" + ("_choose" if args.mode == "choose" else "")
df.to_csv(f"conf/{args.set}_llm_{TAG}.csv", index=False)


def wer(sub, col):
    return sub[col].sum() / max(sub["n_words"].sum(), 1)


# per tier
print(f"\n===== {args.set}  ({TAG}) =====")
tier_rows = []
for t in ["high", "medium", "low"]:
    s = df[df["tier"] == t]
    if len(s):
        tier_rows.append({"tier": t, "N": len(s), "WER before": wer(s, "err_before"),
                          "WER after LLM": wer(s, "err_after"),
                          "improved": int((s["err_after"] < s["err_before"]).sum()),
                          "made worse": int((s["err_after"] > s["err_before"]).sum())})
tt = pd.DataFrame(tier_rows)
print(tt.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# strategies
df["err_medium_only"] = np.where(df["tier"] == "medium", df["err_after"], df["err_before"])
kept = df[df["tier"] != "low"]   # what the robot actually uses (low = ask to repeat)
strat = pd.DataFrame([
    {"strategy": "none", "WER all": wer(df, "err_before"), "WER robot (no low)": wer(kept, "err_before")},
    {"strategy": "medium only", "WER all": wer(df, "err_medium_only"), "WER robot (no low)": wer(kept, "err_medium_only")},
    {"strategy": "all", "WER all": wer(df, "err_after"), "WER robot (no low)": wer(kept, "err_after")},
])
print("\n" + strat.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nfallback used: {df['fallback'].mean():.1%}   "
      f"LLM time per sentence: {df['llm_sec'][df['llm_sec'] > 0].mean():.2f}s   total {time.time() - t_all:.0f}s")
pd.concat([tt.assign(part="tier"), strat.assign(part="strategy")]).to_csv(
    f"figs/llm_summary_{args.set}_{TAG}.csv", index=False)

# best examples for slides
ex = df[(df["tier"] == "medium") & (df["err_after"] < df["err_before"])].copy()
ex["gain"] = ex["err_before"] - ex["err_after"]
print("\nGOOD EXAMPLES (medium tier):")
for _, r in ex.sort_values("gain", ascending=False).head(3).iterrows():
    print(f"   REF: {r['ref']}\n   T1 : {r['hyp']}\n   LLM: {r['llm']}\n")
